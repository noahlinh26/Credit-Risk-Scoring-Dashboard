import warnings
warnings.filterwarnings('ignore')

import pandas as pd
df=pd.read_csv('credit_risk_dataset.csv')

print("Kích thước dữ liệu:", df.shape)
print("Tổng quan kiểu dữ liệu:")
print(df.info())
print("\nSố dòng bị khuyết thiếu dữ liệu:")
missing_data=df.isnull().sum()
print(missing_data[missing_data>0])

#Kiểm tra tỷ lệ nợ xấu của ngân hàng
print(df['loan_status'].value_counts())
print("\nTỷ lệ nợ xấu:")
print(df['loan_status'].value_counts(normalize=True)*100) #normalize=True để chuyển số lượng thành dạng thập phân sau đó nhân với 100 để ra tỷ lệ

#Tìm các giá trị ngoại lai 
import matplotlib.pyplot as plt
import seaborn as sns

#xem các chỉ số thống kế cơ bản của các biến số
print(df[['person_age', 'person_income', 'person_emp_length', 'loan_amnt']].describe()) #describe thống kê các cột chứ dữ liệu số

#Vẽ biểu đồ hộp (boxplot) để phát hiện ngoại lai 
fig, axes = plt.subplots(1, 4, figsize=(18,5))
#vẽ boxplot cho cột person_age
sns.boxplot(x=df['person_age'], ax=axes[0], color='lightblue')
axes[0].set_title('Outlier of Person Age')
#Vẽ biểu đồ hộp (boxplot) để phát hiện ngoại lai của cột Thu nhập
sns.boxplot(x=df['person_income'], ax=axes[1], color='salmon')
axes[1].set_title('Outlier of Person Imcome')
#Vẽ biểu đồ hộp (boxplot) để phát hiện ngoại lai của cột Số năm làm việc
sns.boxplot(x=df['person_emp_length'], ax=axes[2], color='lightgreen')
axes[2].set_title('Outlier of Person Emp Length')
#Vẽ biểu đồ hộp (boxplot) để phát hiện ngoại lai của cột Số tiền vay
sns.boxplot(x=df['loan_amnt'], ax=axes[3], color='lightyellow')
axes[3].set_title('Outlier of Loan Amount')
plt.tight_layout() #tight_layout là để tự động điều chỉnh khoảng cách giữa các biểu đồ
plt.show() #show() là để hiển thị biểu đồ

#Xem tỷ lệ nợ xấu theo hình thức sở hữu nhà ở (Rent, Own, Mortgage)
print("\nTỷ lệ nợ xấu theo hình thức sở hữu nhà ở:")
print(df.groupby('person_home_ownership')['loan_status'].mean()*100) 
#Xem tỷ lệ nợ xấu theo mục đích vay
print("\nTỷ lệ nợ xấu theo mục đích vay:")
print(df.groupby('loan_intent')['loan_status'].mean()*100)
#Xem tỷ lệ nợ xấu theo xếp hạng khoản vay
print("\n Tỷ lệ nợ xấu theo xếp hạng khoản vay:")
print(df.groupby('loan_grade')['loan_status'].mean()*100)
#Xem tỷ lệ nợ xấu theo lịch sử tín dụng
print("\n Tỷ lệ nợ xấu theo lịch sử tín dụng:")
print(df.groupby('cb_person_default_on_file')['loan_status'].mean()*100)
#so sánh các chỉ số tài chính trung bình giữa nhóm trả nợ tốt và trả nợ xấu
num_cols=['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
print(df.groupby('loan_status')[num_cols].mean().T) 

from sklearn.model_selection import train_test_split #thư viện này dùng để chia dữ liệu thành train và test

#copy dataframe ban đầu thành 1 bản mới để clean
df_clean=df.copy()

#Lọc Outlier phi lý
df_clean=df_clean[(df_clean['person_age']>18) & (df_clean['person_age']<=100)] #lọc những người có tuổi từ 18 đến 100
df_clean=df_clean[df_clean['person_emp_length']<=50] #lọc những người có số năm làm việc <=50
print("\n Kích thước dữ liệu sau khi lọc outlier:", df_clean.shape)

#Xử lý dữ liệu khuyết thiếu
#điển các ô trống của cột person_emp_length và loan_int_rate bằng giá trị trung vị của cột đó
#fillna là hàm để điền giá trị khuyết thiếu, median là hàm để tính giá trị trung vị
median_emp=df_clean['person_emp_length'].median()
df_clean['person_emp_length']=df_clean['person_emp_length'].fillna(median_emp)
median_loan=df_clean['loan_int_rate'].median()
df_clean['loan_int_rate']=df_clean['loan_int_rate'].fillna(median_loan)
print("Số dòng bị khuyết thiếu sau khi điền giá trị trung vị:", df_clean.isnull().sum().sum()) #sum().sum() để tính tổng số dòng bị khuyết thiếu, kq phải bằng 0

#Mã hóa dự liệu dạng chữ thành dạng số 0 và 1
#get_dummies là hàm để mã hóa dữ liệu dạng chữ thành dạng số 0 và 1
df_encoded=pd.get_dummies(df_clean,drop_first=True) #drop_first=True để bỏ cột đầu tiên của mỗi biến phân loại, tránh đa cộng tuyến

#Tách tập dữ liệu thành Train và Test
#x: tập dữ liệu đầu vào, y: tập dữ liệu đầu ra (biến mục tiêu loan_status)
x=df_encoded.drop('loan_status', axis=1) #drop cột loan_status vì đây là cột đầu ra, axis=1 để chỉ định xóa cột
y=df_encoded['loan_status']
#Tách 80% dữ liệu để Train và 20% dữ liệu để Test
#random_state=42 để đảm bảo kết quả chia dữ liệu giống nhau mỗi lần chạy
#test_size=0.2 để chia 20% dữ liệu để Test
#stratify=y để đảm bảo tỷ lệ nợ xấu trong tập Train và Test giống nhau
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) #train_test_split là hàm để chia dữ liệu thành train và test
print("\n ----Kết quả tách dữ liệu ----")
print(f"Kích thước tập Train: {x_train.shape}")
print(f"Kích thước tập Test: {x_test.shape}")

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score 
#AUC là chỉ số đánh giá mô hình phân loại, càng gần 1 càng tốt
#ROC là đường cong thể hiện mối quan hệ giữa tỷ lệ dương tính giả và tỷ lệ dương tính thật của mô hình phân loại

#Khởi tạo và huấn luyện dữ liệu
models={
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), #max_iter là số lần lặp tối đa để tìm nghiệm tốt nhất
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=100 là số lượng cây được quyết định có trong rừng
}
print("\n ----Kết quả đánh giá mô hình ----")

#Train và Test AI
for name, model in models.items():
    #Cho mô hình học trên tập Train
    model.fit(x_train, y_train) #đọc nhãn x_train và đối chiếu với y_train để tìm ra quy luật người bùng nợ
    #Dự đoán trên tập Test
    y_pred=model.predict(x_test) 
    #bắt mô hình dự đoán nhãn 0 hoặc 1 cho tập dữ liệu x_test
    #lấy ra xác suất bùng nợ (nhãn 1) thay vì chỉ trả về 0 hoặc 1 
    y_pred_proba=model.predict_proba(x_test)[:,1] #predict_proba là hàm để dự đoán xác suất, [:,1] là lấy cột xác suất của lớp 1 (nợ xấu)
    #Tính toán các chỉ số đánh giá mô hình
    #so sánh kết quả AI dự đoán (y_pred/y_pred_proba) với kết quả thực tế (y_test) để tính toán độ chính xác và điểm phân tách rủi ro
    acc=accuracy_score(y_test, y_pred) #accuracy_score là hàm để tính độ chính xác
    auc=roc_auc_score(y_test, y_pred_proba) #roc_auc_score là hàm để tính AUC

    print(f"\n {name.upper()}")
    print(f"Accuracy (Độ chính xác tổng thể): {acc:.4f}") #.4f là để làm tròn số thập phân đến 4 chữ số
    print(f"ROC-AUC Score (Khả năng phân tách rủi ro: {auc:.4f})")
    print("\n Classification Report (Báo cáo phân loại): ")
    print(classification_report(y_test, y_pred)) #classification_report là hàm để in ra báo cáo phân loại

#Trích xuất mô hình Random Forest đã train
rf_model=models["Random Forest"]

#lấy độ quan trọng của từng thuộc tính
importances=rf_model.feature_importances_ #feature_importances_ tự động tính toán xem mỗi cột dữ liệu đóng góp bao nhiêu % 
feature_name=x.columns #lấy ra danh sách tên của các cột đặc trưng đã được xác định ở trên

#tạo dataframe để dễ sắp xếp
df_importances=pd.DataFrame({
    "Feature":feature_name,
    "Importances":importances
    }).sort_values(by='Importances', ascending=False) #sắp xếp cột dữ liệu theo chiều giảm dần theo sức ảnh hưởng của các cột dữ liệu

print("\n Top 10 cột quan trọng nhất với Random Forest:")
print(df_importances.head(10))

from sklearn.metrics import confusion_matrix
#Vẽ ma trận nhầm lẫn (confusion matrix) để đánh giá mô hình
cm=confusion_matrix(y_test,y_pred) #confusion_matrix là hàm để tính ma trận nhầm lẫn
print(f"Thực tế 0, Dự đoán 0 (Trả nợ tốt đúng): {cm[0,0]}")
print(f"Thực tế 0, Dự đoán 1 (Báo động nhầm): {cm[0,1]}")
print(f"Thực tế 1, Dự đoán 0 (Bỏ sót nợ xấu): {cm[1,0]}")
print(f"Thực tế 1, Dự đoán 1 (Bắt đúng nợ xấu): {cm[1,1]}")

df_clean.to_csv('credit_risk_dataset_cleaned.csv', index=False) #lưu dữ liệu đã được làm sạch ra file csv