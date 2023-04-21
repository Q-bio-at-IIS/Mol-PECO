cd C:/Users/mengji.DESKTOP-U4SLS3J/Desktop/mengji_codes/scentAI/codes/baselines
CALL conda.bat activate pytorch

rem start python run_fgs.py --fp bfps --model svm --data_dir ../../pyrfume_models3_sois/dumped30_coulomb_frobenius/
rem start python run_fgs.py --fp cfps --model svm --data_dir ../../pyrfume_models3_sois/dumped30_coulomb_frobenius/
rem start python run_fgs.py --fp mordreds --model svm --data_dir ../../pyrfume_models3_sois/dumped30_coulomb_frobenius/

start python run_fgs.py --fp mordreds --model smote-gb --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp bfps --model smote-gb --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp cfps --model smote-gb --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp mordreds --model smote-knn --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp bfps --model smote-knn --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp cfps --model smote-knn --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp mordreds --model smote-rf --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp bfps --model smote-rf --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/
start python run_fgs.py --fp cfps --model smote-rf --data_dir ../../pyrfume_sois_canon/dumped30_coulomb_frobenius/

pause