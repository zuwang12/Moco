now=$(date +"%F_%T")

nohup python save_csv.py > "./logs/csv/save_csv_${now}.log" 2>&1 &