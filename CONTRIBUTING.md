# Đóng góp / Contributing

Mở issue mô tả lỗi hoặc pull request với thay đổi nhỏ, rõ mục đích. Dùng Python
3.11 và chạy `python -m pytest -q` khi không đặt `OPENROUTER_API_KEY`.
Với lỗi parser, hãy tạo workbook tổng hợp tối thiểu trong thư mục tạm của test.

Không đưa dữ liệu cuộc thi, đáp án chuẩn, khóa API, prompt hoặc log chứa dữ liệu
riêng vào repository. Các test phụ thuộc tài nguyên ngoài phải skip khi thiếu.
Cập nhật tài liệu khi thay đổi giao diện hoặc cấu hình.

Please open an issue or a focused pull request, explain the change, and run the
offline tests. Use minimal synthetic fixtures. Never submit contest datasets,
gold answers, credentials, or private runtime artifacts. Keep documentation in
sync with behavior and skip optional external fixtures when absent.
