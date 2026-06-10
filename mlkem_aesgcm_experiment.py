import gc
import os
import time
import platform
import statistics
import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REPEAT_TIMES = 100

WARMUP_TIMES = 10


PLAINTEXT_SIZES = {
    "1KB": 1024,
    "100KB": 100 * 1024,
    "1MB": 1024 * 1024,
}

AES_GCM_NONCE_SIZE = 12

# AES-GCM 认证标签长度通常为 16 字节
AES_GCM_TAG_SIZE = 16

HKDF_INFO = b"ML-KEM + AES-GCM experimental key derivation"

RAW_OUTPUT_CSV = "mlkem_aesgcm_raw_results.csv"
SUMMARY_OUTPUT_CSV = "mlkem_aesgcm_summary_results.csv"
COMM_OUTPUT_CSV = "mlkem_aesgcm_communication_overhead.csv"
CORRECTNESS_OUTPUT_CSV = "mlkem_aesgcm_correctness_results.csv"


def load_kem_module(candidate_names: List[str]):

    errors = []
    for name in candidate_names:
        try:
            return importlib.import_module(name)
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise ImportError(
        "无法导入对应的 ML-KEM/Kyber 模块。\n"
        "请确认已安装 pqcrypto：pip install pqcrypto\n"
        "尝试过的模块如下：\n" + "\n".join(errors)
    )


def load_all_kems() -> Dict[str, Any]:
    kem_candidates = {
        "ML-KEM-512": [
            "pqcrypto.kem.ml_kem_512",
            "pqcrypto.kem.kyber512",
        ],
        "ML-KEM-768": [
            "pqcrypto.kem.ml_kem_768",
            "pqcrypto.kem.kyber768",
        ],
        "ML-KEM-1024": [
            "pqcrypto.kem.ml_kem_1024",
            "pqcrypto.kem.kyber1024",
        ],
    }

    modules = {}
    for alg_name, candidates in kem_candidates.items():
        modules[alg_name] = load_kem_module(candidates)

    return modules


# 三、工具函数

def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000


def derive_aes_key(shared_secret: bytes) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


def percentile(values: List[float], p: float) -> float:

    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


@dataclass
class TrialResult:
    algorithm: str
    plaintext_label: str
    plaintext_size_bytes: int
    trial_index: int

    public_key_bytes: int
    secret_key_bytes: int
    kem_ciphertext_bytes: int
    shared_secret_bytes: int

    aes_nonce_bytes: int
    aes_ciphertext_and_tag_bytes: int
    aes_tag_bytes: int
    total_transmission_with_pk_bytes: int
    total_transmission_without_pk_bytes: int

    keygen_ms: float
    encaps_ms: float
    decaps_ms: float
    kdf_alice_ms: float
    kdf_bob_ms: float
    aes_encrypt_ms: float
    aes_decrypt_ms: float

    kem_handshake_ms: float
    kdf_total_ms: float
    aes_total_ms: float
    end_to_end_total_ms: float
    kem_ratio_percent: float

    shared_secret_equal: bool
    plaintext_equal: bool

def run_single_trial(
    kem_module,
    algorithm_name: str,
    plaintext_label: str,
    plaintext: bytes,
    trial_index: int,
) -> TrialResult:


    # ---------- 1. Bob: ML-KEM KeyGen ----------
    t0 = now_ns()
    public_key, secret_key = kem_module.generate_keypair()
    t1 = now_ns()
    keygen_ms = ns_to_ms(t1 - t0)

    # ---------- 2. Alice: ML-KEM Encaps ----------
    t0 = now_ns()
    kem_ciphertext, shared_secret_alice = kem_module.encrypt(public_key)
    t1 = now_ns()
    encaps_ms = ns_to_ms(t1 - t0)

    # ---------- 3. Alice: KDF ----------
    t0 = now_ns()
    aes_key_alice = derive_aes_key(shared_secret_alice)
    t1 = now_ns()
    kdf_alice_ms = ns_to_ms(t1 - t0)

    # ---------- 4. Alice: AES-GCM Encrypt ----------
    aesgcm_alice = AESGCM(aes_key_alice)
    nonce = os.urandom(AES_GCM_NONCE_SIZE)

    t0 = now_ns()
    aes_ciphertext_and_tag = aesgcm_alice.encrypt(
        nonce,
        plaintext,
        associated_data=None
    )
    t1 = now_ns()
    aes_encrypt_ms = ns_to_ms(t1 - t0)

    # ---------- 5. Bob: ML-KEM Decaps ----------
    t0 = now_ns()
    shared_secret_bob = kem_module.decrypt(secret_key, kem_ciphertext)
    t1 = now_ns()
    decaps_ms = ns_to_ms(t1 - t0)

    # ---------- 6. Bob: KDF ----------
    t0 = now_ns()
    aes_key_bob = derive_aes_key(shared_secret_bob)
    t1 = now_ns()
    kdf_bob_ms = ns_to_ms(t1 - t0)

    # ---------- 7. Bob: AES-GCM Decrypt ----------
    aesgcm_bob = AESGCM(aes_key_bob)

    t0 = now_ns()
    decrypted_plaintext = aesgcm_bob.decrypt(
        nonce,
        aes_ciphertext_and_tag,
        associated_data=None
    )
    t1 = now_ns()
    aes_decrypt_ms = ns_to_ms(t1 - t0)

    # ---------- 8. 正确性验证 ----------
    shared_secret_equal = shared_secret_alice == shared_secret_bob
    plaintext_equal = decrypted_plaintext == plaintext

    if not shared_secret_equal:
        raise RuntimeError(f"{algorithm_name} 共享秘密不一致。")

    if not plaintext_equal:
        raise RuntimeError(f"{algorithm_name} AES-GCM 解密明文不一致。")

    # ---------- 9. 通信数据量 ----------
    public_key_bytes = len(public_key)
    secret_key_bytes = len(secret_key)
    kem_ciphertext_bytes = len(kem_ciphertext)
    shared_secret_bytes = len(shared_secret_alice)

    aes_nonce_bytes = len(nonce)
    aes_ciphertext_and_tag_bytes = len(aes_ciphertext_and_tag)

    aes_tag_bytes = AES_GCM_TAG_SIZE

    total_transmission_with_pk_bytes = (
        public_key_bytes
        + kem_ciphertext_bytes
        + aes_nonce_bytes
        + aes_ciphertext_and_tag_bytes
    )

    total_transmission_without_pk_bytes = (
        kem_ciphertext_bytes
        + aes_nonce_bytes
        + aes_ciphertext_and_tag_bytes
    )

    # ---------- 10. 综合指标 ----------
    kem_handshake_ms = keygen_ms + encaps_ms + decaps_ms
    kdf_total_ms = kdf_alice_ms + kdf_bob_ms
    aes_total_ms = aes_encrypt_ms + aes_decrypt_ms
    end_to_end_total_ms = kem_handshake_ms + kdf_total_ms + aes_total_ms

    kem_ratio_percent = (
        kem_handshake_ms / end_to_end_total_ms * 100
        if end_to_end_total_ms > 0
        else 0.0
    )

    return TrialResult(
        algorithm=algorithm_name,
        plaintext_label=plaintext_label,
        plaintext_size_bytes=len(plaintext),
        trial_index=trial_index,

        public_key_bytes=public_key_bytes,
        secret_key_bytes=secret_key_bytes,
        kem_ciphertext_bytes=kem_ciphertext_bytes,
        shared_secret_bytes=shared_secret_bytes,

        aes_nonce_bytes=aes_nonce_bytes,
        aes_ciphertext_and_tag_bytes=aes_ciphertext_and_tag_bytes,
        aes_tag_bytes=aes_tag_bytes,
        total_transmission_with_pk_bytes=total_transmission_with_pk_bytes,
        total_transmission_without_pk_bytes=total_transmission_without_pk_bytes,

        keygen_ms=keygen_ms,
        encaps_ms=encaps_ms,
        decaps_ms=decaps_ms,
        kdf_alice_ms=kdf_alice_ms,
        kdf_bob_ms=kdf_bob_ms,
        aes_encrypt_ms=aes_encrypt_ms,
        aes_decrypt_ms=aes_decrypt_ms,

        kem_handshake_ms=kem_handshake_ms,
        kdf_total_ms=kdf_total_ms,
        aes_total_ms=aes_total_ms,
        end_to_end_total_ms=end_to_end_total_ms,
        kem_ratio_percent=kem_ratio_percent,

        shared_secret_equal=shared_secret_equal,
        plaintext_equal=plaintext_equal,
    )


def warmup(kem_modules: Dict[str, Any]) -> None:

    print(f"开始预热，每个参数集预热 {WARMUP_TIMES} 次...")

    warmup_plaintext = os.urandom(1024)

    for alg_name, module in kem_modules.items():
        for i in range(WARMUP_TIMES):
            run_single_trial(
                kem_module=module,
                algorithm_name=alg_name,
                plaintext_label="warmup",
                plaintext=warmup_plaintext,
                trial_index=i,
            )

    print("预热完成。\n")

def run_experiment() -> pd.DataFrame:

    print("加载 ML-KEM / Kyber 参数集模块...")
    kem_modules = load_all_kems()
    print("模块加载完成：", ", ".join(kem_modules.keys()))
    print()

    print("实验环境信息：")
    print(f"操作系统: {platform.platform()}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"处理器信息: {platform.processor()}")
    print(f"测试次数: {REPEAT_TIMES}")
    print(f"预热次数: {WARMUP_TIMES}")
    print()

    warmup(kem_modules)

    gc.collect()
    gc.disable()

    all_results: List[TrialResult] = []

    try:
        for alg_name, module in kem_modules.items():
            for plaintext_label, size in PLAINTEXT_SIZES.items():
                print(f"正在测试：{alg_name}, 明文大小：{plaintext_label}")

                plaintext = os.urandom(size)

                for i in range(1, REPEAT_TIMES + 1):
                    result = run_single_trial(
                        kem_module=module,
                        algorithm_name=alg_name,
                        plaintext_label=plaintext_label,
                        plaintext=plaintext,
                        trial_index=i,
                    )
                    all_results.append(result)

                print(f"完成：{alg_name}, 明文大小：{plaintext_label}")

    finally:
        gc.enable()

    raw_df = pd.DataFrame([r.__dict__ for r in all_results])
    return raw_df



def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    group_cols = ["algorithm", "plaintext_label", "plaintext_size_bytes"]

    metrics = [
        "keygen_ms",
        "encaps_ms",
        "decaps_ms",
        "kem_handshake_ms",
        "kdf_total_ms",
        "aes_total_ms",
        "end_to_end_total_ms",
        "kem_ratio_percent",
    ]

    for group_key, group in raw_df.groupby(group_cols):
        algorithm, plaintext_label, plaintext_size_bytes = group_key

        row = {
            "参数集": algorithm,
            "明文大小": plaintext_label,
            "明文字节数": plaintext_size_bytes,
            "测试次数": len(group),
        }

        for metric in metrics:
            stats = summarize(group[metric].tolist())
            prefix = metric.replace("_ms", "").replace("_percent", "")
            for stat_name, stat_value in stats.items():
                row[f"{prefix}_{stat_name}"] = stat_value

        row["ML-KEM握手平均耗时/ms"] = row["kem_handshake_mean_ms"]
        row["AES-GCM加解密平均耗时/ms"] = row["aes_total_mean_ms"]
        row["KDF平均耗时/ms"] = row["kdf_total_mean_ms"]
        row["端到端平均总耗时/ms"] = row["end_to_end_total_mean_ms"]
        row["ML-KEM平均开销占比/%"] = row["kem_ratio_mean_ms"]

        row["端到端P95耗时/ms"] = row["end_to_end_total_p95_ms"]
        row["端到端标准差/ms"] = row["end_to_end_total_std_ms"]

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # 按参数集和明文大小排序
    algorithm_order = {
        "ML-KEM-512": 1,
        "ML-KEM-768": 2,
        "ML-KEM-1024": 3,
    }
    size_order = {
        "1KB": 1,
        "100KB": 2,
        "1MB": 3,
    }

    summary_df["alg_order"] = summary_df["参数集"].map(algorithm_order)
    summary_df["size_order"] = summary_df["明文大小"].map(size_order)
    summary_df = summary_df.sort_values(["size_order", "alg_order"])
    summary_df = summary_df.drop(columns=["alg_order", "size_order"])

    return summary_df


def build_communication_overhead(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成通信开销表。
    """
    rows = []

    group_cols = ["algorithm", "plaintext_label", "plaintext_size_bytes"]

    for group_key, group in raw_df.groupby(group_cols):
        algorithm, plaintext_label, plaintext_size_bytes = group_key
        first = group.iloc[0]

        rows.append({
            "参数集": algorithm,
            "明文大小": plaintext_label,
            "明文字节数": plaintext_size_bytes,
            "公钥长度/B": int(first["public_key_bytes"]),
            "私钥长度/B": int(first["secret_key_bytes"]),
            "ML-KEM密文长度/B": int(first["kem_ciphertext_bytes"]),
            "共享秘密长度/B": int(first["shared_secret_bytes"]),
            "AES-GCM nonce长度/B": int(first["aes_nonce_bytes"]),
            "AES-GCM密文与Tag长度/B": int(first["aes_ciphertext_and_tag_bytes"]),
            "AES-GCM Tag长度/B": int(first["aes_tag_bytes"]),
            "计入公钥的总传输量/B": int(first["total_transmission_with_pk_bytes"]),
            "不计公钥的总传输量/B": int(first["total_transmission_without_pk_bytes"]),
        })

    comm_df = pd.DataFrame(rows)

    algorithm_order = {
        "ML-KEM-512": 1,
        "ML-KEM-768": 2,
        "ML-KEM-1024": 3,
    }
    size_order = {
        "1KB": 1,
        "100KB": 2,
        "1MB": 3,
    }

    comm_df["alg_order"] = comm_df["参数集"].map(algorithm_order)
    comm_df["size_order"] = comm_df["明文大小"].map(size_order)
    comm_df = comm_df.sort_values(["size_order", "alg_order"])
    comm_df = comm_df.drop(columns=["alg_order", "size_order"])

    return comm_df


def build_correctness(raw_df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    group_cols = ["algorithm", "plaintext_label", "plaintext_size_bytes"]

    for group_key, group in raw_df.groupby(group_cols):
        algorithm, plaintext_label, plaintext_size_bytes = group_key

        test_count = len(group)
        shared_secret_success = int(group["shared_secret_equal"].sum())
        plaintext_success = int(group["plaintext_equal"].sum())

        rows.append({
            "参数集": algorithm,
            "明文大小": plaintext_label,
            "明文字节数": plaintext_size_bytes,
            "测试次数": test_count,
            "共享秘密一致次数": shared_secret_success,
            "解密成功次数": plaintext_success,
            "共享秘密一致率/%": shared_secret_success / test_count * 100,
            "解密成功率/%": plaintext_success / test_count * 100,
        })

    correctness_df = pd.DataFrame(rows)

    algorithm_order = {
        "ML-KEM-512": 1,
        "ML-KEM-768": 2,
        "ML-KEM-1024": 3,
    }
    size_order = {
        "1KB": 1,
        "100KB": 2,
        "1MB": 3,
    }

    correctness_df["alg_order"] = correctness_df["参数集"].map(algorithm_order)
    correctness_df["size_order"] = correctness_df["明文大小"].map(size_order)
    correctness_df = correctness_df.sort_values(["size_order", "alg_order"])
    correctness_df = correctness_df.drop(columns=["alg_order", "size_order"])

    return correctness_df

def print_paper_tables(
    summary_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    correctness_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 90)
    print("表 A：通信开销表")
    print("=" * 90)

    paper_comm = comm_df[
        [
            "参数集",
            "明文大小",
            "公钥长度/B",
            "ML-KEM密文长度/B",
            "共享秘密长度/B",
            "计入公钥的总传输量/B",
            "不计公钥的总传输量/B",
        ]
    ].copy()

    print(paper_comm.to_string(index=False))

    print("\n" + "=" * 90)
    print("表 B：端到端耗时表")
    print("=" * 90)

    paper_summary = summary_df[
        [
            "参数集",
            "明文大小",
            "ML-KEM握手平均耗时/ms",
            "AES-GCM加解密平均耗时/ms",
            "KDF平均耗时/ms",
            "端到端平均总耗时/ms",
            "ML-KEM平均开销占比/%",
            "端到端P95耗时/ms",
            "端到端标准差/ms",
        ]
    ].copy()

    numeric_cols = paper_summary.select_dtypes(include=["float64", "float32"]).columns
    paper_summary[numeric_cols] = paper_summary[numeric_cols].round(4)

    print(paper_summary.to_string(index=False))

    print("\n" + "=" * 90)
    print("表 C：正确性验证表")
    print("=" * 90)

    paper_correctness = correctness_df[
        [
            "参数集",
            "明文大小",
            "测试次数",
            "共享秘密一致次数",
            "解密成功次数",
            "共享秘密一致率/%",
            "解密成功率/%",
        ]
    ].copy()

    numeric_cols = paper_correctness.select_dtypes(include=["float64", "float32"]).columns
    paper_correctness[numeric_cols] = paper_correctness[numeric_cols].round(2)

    print(paper_correctness.to_string(index=False))


def main() -> None:
    raw_df = run_experiment()

    summary_df = build_summary(raw_df)
    comm_df = build_communication_overhead(raw_df)
    correctness_df = build_correctness(raw_df)

    raw_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    comm_df.to_csv(COMM_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    correctness_df.to_csv(CORRECTNESS_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print_paper_tables(summary_df, comm_df, correctness_df)

    print("\n实验完成，已生成以下文件：")
    print(f"1. 原始实验数据：{RAW_OUTPUT_CSV}")
    print(f"2. 统计汇总数据：{SUMMARY_OUTPUT_CSV}")
    print(f"3. 通信开销数据：{COMM_OUTPUT_CSV}")
    print(f"4. 正确性验证数据：{CORRECTNESS_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
