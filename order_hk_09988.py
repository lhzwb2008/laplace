import time
import json
import pymysql
from futu import *
class OrderBookTest(OrderBookHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(OrderBookTest,self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("OrderBookTest: error, msg: %s" % data)
            return RET_ERROR, data
        if data['svr_recv_time_bid'] == '':
            return RET_ERROR, data
        db = pymysql.connect(host="rm-uf6h8okah66ri878uko.mysql.rds.aliyuncs.com", user="sessionloops", password="Hello2021", database="laplace" )
        cursor = db.cursor()
        sql = "INSERT INTO order_hk_09988(code, svr_recv_time_bid, svr_recv_time_ask, bid, ask) \
        VALUES ('%s', '%s',  '%s',  '%s',  '%s')" % \
        (data['code'], data['svr_recv_time_bid'], data['svr_recv_time_ask'],json.dumps( data['Bid']), json.dumps(data['Ask']))
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
handler = OrderBookTest()
quote_ctx.set_handler(handler)  # 设置实时摆盘回调
quote_ctx.subscribe(['HK.09988'], [SubType.ORDER_BOOK])  # 订阅买卖摆盘类型，FutuOpenD 开始持续收到服务器的推送
