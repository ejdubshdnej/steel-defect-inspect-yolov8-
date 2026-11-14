"""把 ultralytics 数据加载里的 ThreadPool 换成串行实现。

为什么要这么做：Windows 上 ultralytics 的 ThreadPool 在部分环境下会直接崩
（共享内存/管道相关的原生异常，栈里看不到 Python 层的错误），表现为训练或
导出在第一个 batch 之前就退出。把这些并行原语替换成串行版本可以稳定跑通，
代价是数据加载变慢——在单机小数据集上影响有限，是可以接受的折中。

用法：
    import shim; shim.apply()
    from ultralytics import YOLO      # 顺序不能反

关键约束：apply() 必须在 import ultralytics 之前调用。ultralytics 在导入时
就会把 ThreadPool 绑定到模块属性上，之后再替换已经来不及了。

注意：本模块只替换 ultralytics.data.dataset 里的 ThreadPool 引用，不改变
任何数据增强、采样或标注解析的逻辑，输出与并行版本一致。
"""
import ultralytics.data.dataset as D


class SerialPool:
    """ThreadPool / multiprocessing.Pool 的最小串行替身。

    只实现 ultralytics 实际用到的那几个方法（imap/map/starmap 与
    上下文管理），语义上是「按顺序一个个做」，不做任何并行。
    """

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def imap(self, func, iterable, *a, **k):
        for item in iterable:
            yield func(item)
    def map(self, func, iterable, *a, **k):
        return [func(i) for i in iterable]
    def starmap(self, func, iterable, *a, **k):
        return [func(*i) for i in iterable]
    def close(self): pass
    def join(self): pass
    def terminate(self): pass


def apply():
    """把 ultralytics 使用的线程池替换成 SerialPool。必须在导入前调用。"""
    D.ThreadPool = SerialPool
