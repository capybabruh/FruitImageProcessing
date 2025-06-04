# 🍎🍋🌶️ Phân loại Trái cây và Rau củ bằng CNN 🥦🥕🥭

Dự án này sử dụng Mạng nơ-ron tích chập (CNN) được xây dựng bằng TensorFlow/Keras để phân loại hình ảnh các loại trái cây và rau củ khác nhau. Mục tiêu là xác định chính xác các loại nông sản từ hình ảnh.

---

## 🌟 Mục lục

* [Tổng quan](#-tổng-quan)
* [Dữ liệu](#-dữ-liệu)
* [Cấu trúc dự án](#-cấu-trúc-dự-án)
* [Các thư viện chính](#-các-thư-viện-chính)
* [Các bước phân tích](#-các-bước-phân-tích)
    * [1. Tải và chuẩn bị dữ liệu](#1-tải-và-chuẩn-bị-dữ-liệu)
    * [2. Xây dựng mô hình CNN](#2-xây-dựng-mô-hình-cnn)
    * [3. Huấn luyện mô hình](#3-huấn-luyện-mô-hình)
    * [4. Đánh giá và trực quan hóa mô hình](#4-đánh-giá-và-trực-quan-hóa-mô-hình)
    * [5. Dự đoán trên ảnh mới](#5-dự-đoán-trên-ảnh-mới)
* [Cách chạy mã](#-cách-chạy-mã)
* [Kết quả và hiểu biết](#-kết-quả-và-hiểu-biết)

---

## 🚀 Tổng quan

Mô hình được xây dựng để nhận diện 36 loại trái cây và rau củ khác nhau. Đây là một ứng dụng điển hình của thị giác máy tính trong việc tự động hóa quá trình phân loại sản phẩm nông nghiệp.

---

## 🍓 Dữ liệu

Mô hình được huấn luyện và xác thực trên tập dữ liệu hình ảnh trái cây và rau củ, được tổ chức thành các thư mục `train`, `validation` và `test`. Mỗi thư mục con chứa các thư mục được đặt tên theo từng lớp (ví dụ: `apple`, `banana`, `bell pepper`).

**Đường dẫn dữ liệu:** `/content/drive/MyDrive/archive/`

---

## 📁 Cấu trúc dự án

Mã dự án được thiết kế để chạy trong môi trường Google Colab vì nó phụ thuộc vào Google Drive để truy cập tập dữ liệu. Kịch bản chính thực hiện các hành động sau:

* Kết nối Google Drive.
* Tải tập dữ liệu huấn luyện, xác thực và kiểm tra.
* Định nghĩa và biên dịch kiến trúc CNN.
* Huấn luyện CNN.
* Lưu mô hình đã huấn luyện.
* Trực quan hóa lịch sử huấn luyện.
* Minh họa dự đoán trên một hình ảnh duy nhất.

---

## 📚 Các thư viện chính

* `numpy`: Cho các phép toán số học.
* `matplotlib.pyplot`: Để vẽ biểu đồ và hiển thị hình ảnh.
* `tensorflow`: Thư viện cốt lõi để xây dựng và huấn luyện mô hình học sâu.
* `cv2` (OpenCV): Để xử lý hình ảnh, đặc biệt là tải và hiển thị hình ảnh.

---

## 📈 Các bước phân tích

### 1. Tải và chuẩn bị dữ liệu

Các tập dữ liệu hình ảnh cho huấn luyện, xác thực và kiểm tra được tải trực tiếp từ Google Drive bằng cách sử dụng `tf.keras.utils.image_dataset_from_directory`. Tiện ích này tự động suy ra nhãn lớp từ tên thư mục và chuẩn bị hình ảnh theo từng batch.

* **Kích thước hình ảnh:** Tất cả hình ảnh được thay đổi kích thước thành `64x64` pixel.
* **Chế độ màu:** Hình ảnh được xử lý ở chế độ màu `rgb`.
* **Chế độ nhãn:** Nhãn được đặt là `categorical` cho phân loại đa lớp.
* **Kích thước batch:** Dữ liệu được tải theo các batch có kích thước `32` hình ảnh.

### 2. Xây dựng mô hình CNN

Kiến trúc CNN được thiết kế bằng cách sử dụng `tf.keras.models.Sequential`, một chồng các lớp tuyến tính.

Mô hình bao gồm:

* **Các lớp tích chập (`Conv2D`):**
    * Hai khối lớp `Conv2D` (32 bộ lọc, kích thước kernel 3x3) theo sau là kích hoạt `relu`, chịu trách nhiệm trích xuất đặc trưng.
    * Hai khối lớp `Conv2D` khác (64 bộ lọc, kích thước kernel 3x3) với kích hoạt `relu` để học các đặc trưng sâu hơn.
* **Các lớp gộp (`MaxPool2D`):**
    * Các lớp `MaxPool2D` (kích thước pool 2x2, bước nhảy 2x2) được sử dụng sau mỗi khối `Conv2D` để giảm kích thước không gian và trích xuất các đặc trưng nổi bật.
* **Các lớp Dropout (`Dropout`):**
    * Các lớp `Dropout` được đặt một cách chiến lược (`0.25` sau khối gộp thứ hai, và `0.5` sau các lớp flatten và dense) để **ngăn chặn overfitting** bằng cách ngẫu nhiên đặt một phần các đơn vị đầu vào thành 0 trong quá trình huấn luyện.
* **Lớp làm phẳng (`Flatten`):**
    * Chuyển đổi các bản đồ đặc trưng 2D thành một vector 1D để đưa vào các lớp kết nối đầy đủ.
* **Các lớp dày đặc (`Dense`):**
    * Hai lớp `Dense` được kết nối đầy đủ (`512` và `256` đơn vị) với kích hoạt `relu` để học các mẫu cấp cao.
* **Lớp đầu ra (`Dense`):**
    * Một lớp `Dense` cuối cùng với `36` đơn vị (tương ứng với số lượng lớp) và hàm kích hoạt `softmax`, xuất ra xác suất cho mỗi lớp.

### 3. Huấn luyện mô hình

CNN được **biên dịch** với:

* **Bộ tối ưu hóa:** `rmsprop` (Root Mean Square Propagation), một thuật toán tối ưu hóa tốc độ học thích nghi.
* **Hàm mất mát:** `categorical_crossentropy`, phù hợp cho phân loại đa lớp nơi nhãn được mã hóa one-hot.
* **Các chỉ số:** `accuracy`, để theo dõi hiệu suất của mô hình trong quá trình huấn luyện.

Mô hình sau đó được **huấn luyện** bằng phương thức `fit` trên `training_set` và được xác thực trên `validation_set` trong `32` epoch. Lịch sử huấn luyện (độ chính xác và mất mát cho cả tập huấn luyện và xác thực) được ghi lại và lưu vào một file JSON.

### 4. Đánh giá và trực quan hóa mô hình

* **Độ chính xác xác thực cuối cùng** được in ra để đánh giá hiệu suất của mô hình trên dữ liệu chưa thấy.
* Các **biểu đồ** được tạo ra để trực quan hóa:
    * **Độ chính xác huấn luyện so với số epoch:** Cho thấy mức độ học của mô hình trên dữ liệu huấn luyện theo thời gian.
    * **Độ chính xác xác thực so với số epoch:** Cho thấy khả năng khái quát hóa của mô hình và giúp phát hiện overfitting.
    * Tương tự cho **Hàm mất mát huấn luyện và xác thực**.

### 5. Dự đoán trên ảnh mới

Mô hình đã huấn luyện được tải từ `trained_model.h5`. Một minh họa được cung cấp về cách dự đoán lớp của một hình ảnh mới duy nhất:

1.  Một hình ảnh được tải bằng `cv2.imread` và hiển thị.
2.  Hình ảnh sau đó được tiền xử lý (`tf.keras.preprocessing.image.load_img`, `img_to_array`, và `np.array`) để khớp với định dạng đầu vào mà CNN mong đợi.
3.  Phương thức `predict` của mô hình xuất ra điểm xác suất cho mỗi lớp.
4.  Lớp có xác suất cao nhất được xác định là dự đoán, và tên lớp tương ứng được in ra.

---

## 🏃‍♀️ Cách chạy mã

1.  **Kết nối Google Drive:** Mã bắt đầu bằng cách kết nối Google Drive của bạn. Đảm bảo tập dữ liệu có thể truy cập được tại `/content/drive/MyDrive/archive/`.
2.  **Kiểm tra cấu trúc tập dữ liệu:** Xác minh rằng thư mục `archive` của bạn chứa các thư mục con `train`, `validation` và `test`, với các thư mục con cụ thể theo lớp bên trong (ví dụ: `/archive/train/apple/`, `/archive/test/bell pepper/`).
3.  **Chạy trong Google Colab:** Thực thi mã Python được cung cấp trong một sổ ghi chép Google Colab.
4.  **Kiểm tra phụ thuộc:** Tất cả các thư viện cần thiết đều là tiêu chuẩn trong Colab. Nếu không, hãy cài đặt chúng bằng cách sử dụng `pip install <tên_thư_viện>`.

---

## 📊 Kết quả và hiểu biết

Lịch sử huấn luyện cung cấp một cái nhìn tổng quan rõ ràng về quá trình học của mô hình. Như đã quan sát từ các biểu đồ, độ chính xác huấn luyện tăng đều đặn, và độ chính xác xác thực cũng cho thấy sự cải thiện mạnh mẽ, cho thấy mô hình đang học hiệu quả và khái quát hóa tốt trên dữ liệu mới. Tỷ lệ phần trăm độ chính xác xác thực cuối cùng thể hiện khả năng của mô hình trong việc phân loại hình ảnh trái cây và rau củ.

Ví dụ, mô hình đạt được **độ chính xác xác thực ~96.3%** trong lần chạy được cung cấp.
