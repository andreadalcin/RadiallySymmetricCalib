import logging
from visdom import Visdom

try:
    VIS = Visdom(port=8099, raise_exceptions=True)
except ConnectionError:
    VIS = None

class VisdomHandler(logging.Handler):
    """
    A handler class which allows to emit results to visdom.
    """
    max_msgs = 100

    def __init__(self, vis: Visdom):
        logging.Handler.__init__(self)
        assert vis is not None
        self.vis = vis
        self.msgs_count = {ln:0 for ln in list(logging._nameToLevel.keys()) }
        self.vis.close(win=None)

    def emit(self, record):
        try:
            msg = self.format(record)
            level_name = record.levelname
            self.msgs_count[level_name] += 1
            res = self.vis.text(win=level_name,text=msg,append=True, opts={'title': level_name})

            if res != level_name or self.msgs_count[level_name] > self.max_msgs:
                self.msgs_count[level_name] = 0
                self.vis.text(win=level_name,text=msg, opts={'title': level_name})
        except Exception:
            self.handleError(record)

vis_handler = logging.NullHandler() if VIS is None else VisdomHandler(VIS)

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.WARNING)

vis_handler.setFormatter(formatter)
vis_handler.setLevel(logging.DEBUG)

LOGGER = logging.getLogger("bmk")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)
LOGGER.addHandler(vis_handler)
LOGGER.propagate = False

