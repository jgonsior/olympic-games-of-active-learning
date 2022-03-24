# really dirty hack to provide logging as functions instead of objects
import datetime
import threading

logfile_path: str


def init_logger(logfilepath: str) -> None:
    global logfile_path
    logfile_path = logfilepath


def log_it(message: str) -> None:
    global logfile_path
    message = (
        "["
        + str(threading.get_ident())
        + "] ["
        + str(datetime.datetime.now())
        + "] "
        + str(message)
    )

    if logfile_path == "console":
        print(message)
    else:
        with open(logfile_path, "a") as f:
            f.write(message + "\n")
