from trade import Trade


class TradeHistory:
    """Turning the Trading history into an object. This way, the trade information is more legible."""

    def __init__(self, lst=None):
        self.allTrades = lst if lst else [["List of Trades", None, None, None, None, None]]

    def first_trade(self):
        if len(self.allTrades) > 2:
            return Trade(self.allTrades[1])

    def last_trade(self):
        return Trade(self.allTrades[-1])

    def append(self, list_object) -> bool:
        self.allTrades.append(Trade(list_object))
        return True

    def convert_list(lst):
        new_list = TradeHistory()
        new_list.allTrades = lst
        return new_list

    def __len__(self):
        return len(self.allTrades)

    def __getitem__(self, item):
        return self.allTrades[item]

    def __repr__(self):
        print("[", end="")
        for i in self.allTrades[:-1]:
            print(f"{i},")
        print(f"{self.allTrades[-1]}]", end="")
        return ""
