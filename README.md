# LLM-self-built: Reasoning LLM from Scratch

Hệ thống Mô hình Ngôn ngữ Lớn (LLM) tự xây dựng từ đầu (*from scratch*) bằng PyTorch, tối ưu hóa cho huấn luyện trên đơn GPU (Google Colab T4 / RTX 3060 / RTX 4090) kết hợp cơ chế suy luận chuyên sâu Chain-of-Thought (CoT) và Test-Time Compute.

## 📚 Tài Liệu Hướng Dẫn & Kế Hoạch

- **Lộ trình từng bước chi tiết**: [ROADMAP.md](ROADMAP.md)
- **Đặc tả kiến trúc hệ thống**: [specs/001-llm-reasoning-system/spec.md](specs/001-llm-reasoning-system/spec.md)
- **Kế hoạch triển khai Core LLM & KV Cache**: [docs/superpowers/plans/2026-11-10-core-llm-kv-cache.md](docs/superpowers/plans/2026-11-10-core-llm-kv-cache.md)
- **Kế hoạch triển khai Track C (Reasoning & Scaling)**: [docs/superpowers/plans/2026-11-10-track-c-reasoning-scaling.md](docs/superpowers/plans/2026-11-10-track-c-reasoning-scaling.md)

## 🏗️ Cấu Hình Phần Cứng Mục Tiêu

| Cấu hình | Tham số | VRAM Huấn luyện | Phần cứng phù hợp |
|---|---|---|---|
| **Compact Profile** | **~43.8M** | ~5.5 – 7.5 GB | Google Colab T4 (16GB), RTX 3060 (12GB) |
| **Standard Profile** | **~197.2M** | ~12 – 16 GB | Google Colab L4 (24GB), RTX 4090 (24GB) |
