# ML-KEM-
"""
实验内容：
1. 使用 ML-KEM-512 / ML-KEM-768 / ML-KEM-1024 建立共享秘密；
2. 使用 HKDF-SHA256 从共享秘密派生 AES-GCM 会话密钥；
3. 使用 AES-GCM 加密和解密不同大小明文；
4. 记录 ML-KEM 握手耗时、KDF 耗时、AES-GCM 加解密耗时、端到端总耗时；
5. 验证共享秘密一致性和明文解密正确性；
6. 导出原始实验数据和统计汇总数据。

依赖：
pip install pqcrypto cryptography pandas
"""
