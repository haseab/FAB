class Trade:
    """Turning a trade into an object. This way, the trade information is more legible."""

    def __init__(self, trade_info):
        self.trade_info = trade_info
        self.side = trade_info[0] if len(trade_info) == 5 else None
        self.status = trade_info[1] if len(trade_info) == 5 else None
        self.datetime = trade_info[2] if len(trade_info) == 5 else None
        self.price = trade_info[3] if len(trade_info) == 5 else None
        self.rule = trade_info[4] if len(trade_info) == 5 else None

    def __len__(self):
        return len(self.trade_info)

    def __getitem__(self, item):
        return self.trade_info[item]

    def __repr__(self):
        return str(self.trade_info)
