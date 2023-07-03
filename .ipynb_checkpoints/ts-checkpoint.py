import tushare as ts
ts.set_token('b0030b7297c6b297f1db549c7f8c8a080b5f28e2267f07c4741e69ad')
#获取股票1分钟数据
df = ts.pro_bar(ts_code='002594.SZ',freq='1min', start_date='2023-01-01 09:00:00', end_date='2023-01-04 14:13:00')
print(df)
for index,row in df.iterrows(): 
    db = pymysql.connect(host="127.0.0.1", user="root", password="123", database="laplace" )
    cursor = db.cursor()
    sql = "INSERT INTO trade(ts_code,trade_time,vol,amount,price) \
    VALUES ('%s','%s','%s','%s','%s')" % \
    (row['ts_code'],row['trade_time'],row['vol'], row['amount'],row['open'])
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print(e)
    finally:
        db.close()      
