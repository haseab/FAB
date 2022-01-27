class Trade:
    """Turning a trade into an object. This way, the trade information is more legible."""

    def __init__(self, trade_info):
        self.trade_info = trade_info
        if len(trade_info) != 6:
            raise Exception('Trade List is not in proper format!', trade_info)
        self.index = trade_info[0] 
        self.side = trade_info[1] 
        self.status = trade_info[2] 
        self.datetime = trade_info[3] 
        self.price = trade_info[4]
        self.rule = trade_info[5] 

    def __len__(self):
        return len(self.trade_info)

    def __getitem__(self, item):
        return self.trade_info[item]

    def __repr__(self):
        return str(self.trade_info)
