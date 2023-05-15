import time
import json
import pymysql
from futu import *
class TickerTest(TickerHandlerBase):
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
