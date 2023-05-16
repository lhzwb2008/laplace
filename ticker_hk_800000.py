import time
import json
import pymysql
from futu import *

def throttle(min_interval):
    """
    Decorator that prevents a function from being called if it was called less than
    the given amount of time ago.
    """
    def decorator(func):
        func.last_called = 0

        def wrapper(*args, **kwargs):
            now = time.time()
            if now - func.last_called >= min_interval:
                result = func(*args, **kwargs)
                func.last_called = now
                return result
        return wrapper
    return decorator

class TickerTest(TickerHandlerBase):
    @throttle(60)
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(TickerTest,self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("TickerTest: error, msg: %s" % data)
            return RET_ERROR, data
        for index,row in data.iterrows():
            db = pymysql.connect(host="rm-uf6h8okah66ri878uko.mysql.rds.aliyuncs.com", user="sessionloops", password="Hello2021", database="laplace" )
            cursor = db.cursor()
            sql = "INSERT INTO ticker_hk_800000(code,sequence,time,price,volume,data) \
            VALUES ('%s','%s','%s','%s','%s','%s')" % \
            (row['code'],row['sequence'],row['time'], row['price'],row['volume'],str(row))
            try:
                print(sql)
                cursor.execute(sql)
                db.commit()
            except error:
                print(error)
                db.rollback()
            db.close()
        return RET_OK, data
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
handler = TickerTest()
quote_ctx.set_handler(handler) 
quote_ctx.subscribe(['HK.800000'], [SubType.TICKER])
