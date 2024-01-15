import logging
import time
import colorlog


# log_colors_config = {
#     'DEBUG': 'white',  # cyan white
#     'INFO': 'green',
#     'WARNING': 'yellow',
#     'ERROR': 'red',
#     'CRITICAL': 'bold_red',
# }

class LoggerHandler(logging.Logger):

    def __init__(self,
                 name="root",
                 level="DEBUG",
                 file=None,
                 format="%(filename)s:%(lineno)d - %(asctime)s : %(levelname)s - %(message)s"
                 ):
        super().__init__(name)

        self.setLevel(level)  # 设置收集器级别

        # # 设置颜色
        # console_formatter = colorlog.ColoredFormatter(
        #     fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] '
        #         ': %(message)s',
        #     datefmt='%Y-%m-%d  %H:%M:%S',
        #     log_colors=log_colors_config
        # )
        # console_handler.setFormatter(console_formatter)
        # file_handler.setFormatter(file_formatter)

        fmt = logging.Formatter(format)  # 初始化format，设置格式

        # 如果file为空，就执行stream_handler,如果有，两个都执行
        if file:
            file_handler = logging.FileHandler(file)
            # 设置handler级别
            file_handler.setLevel(level)
            # 添加handler
            self.addHandler(file_handler)
            # 添加日志处理器
            file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        self.addHandler(stream_handler)
        stream_handler.setFormatter(fmt)


# 为了确保每次是同一个文件，调用同样的logger对象(防止手误写错文件名字),所以在这里直接初始化logger这个对象比较好
# 可以将name,file参数写入配置文件中（这里我是直接写到了配置文件当中，也可以直接传）
init_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
logger = LoggerHandler("RootLog", file="../logs/{}.txt".format(init_time))

if __name__ == '__main__':
    logger.info("aaa")
    logger.info("info")
    logger.debug("debug")
    logger.warning("warning")
    logger.error("error")
