import time
import functools
import os

# 全局注册表存储所有被装饰的函数
_profiled_functions = {}
# 模块级变量，用于跟踪当前运行中是否已经初始化过文件
_performance_file_initialized = False

def profile_operation(func):
    """装饰器用于测量函数的执行时间和调用次数"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 记录性能数据
        wrapper.call_count += 1
        wrapper.total_time += execution_time
        wrapper.max_time = max(wrapper.max_time, execution_time)
        wrapper.min_time = min(wrapper.min_time, execution_time) if wrapper.call_count > 1 else execution_time
        
        # 获取函数显示名称（如果是方法，包含类名）
        display_name = _get_function_display_name(func, args)
        
        # 注册到全局字典
        _profiled_functions[display_name] = wrapper
        
        # 每100次调用打印一次统计信息（避免输出过多）
        if wrapper.call_count % 100 == 0:
            avg_time = wrapper.total_time / wrapper.call_count
            print(f"[PROFILE] {display_name}: calls={wrapper.call_count}, "
                  f"avg={avg_time:.3f}ms, max={wrapper.max_time:.3f}ms, "
                  f"min={wrapper.min_time:.3f}ms, total={wrapper.total_time:.3f}ms")
        
        return result
    
    # 初始化性能统计变量
    wrapper.call_count = 0
    wrapper.total_time = 0.0
    wrapper.max_time = 0.0
    wrapper.min_time = float('inf')
    
    return wrapper

def _get_function_display_name(func, args):
    """获取函数的显示名称，如果是方法则包含类名"""
    # 检查是否是实例方法（第一个参数是self）
    if args and hasattr(args[0], '__class__'):
        class_name = args[0].__class__.__name__
        return f"{class_name}.{func.__name__}"
    # 检查是否是类方法
    elif hasattr(func, '__self__') and func.__self__ is not None:
        class_name = func.__self__.__class__.__name__
        return f"{class_name}.{func.__name__}"
    else:
        return func.__name__

def get_performance_stats():
    """获取所有被装饰函数的性能统计"""
    stats = {}
    for name, wrapper in _profiled_functions.items():
        if wrapper.call_count > 0:
            stats[name] = {
                'calls': wrapper.call_count,
                'total_time_ms': wrapper.total_time,
                'avg_time_ms': wrapper.total_time / wrapper.call_count,
                'max_time_ms': wrapper.max_time,
                'min_time_ms': wrapper.min_time
            }
    return stats

def print_performance_summary():
    """将性能摘要写入文件，每次运行清空，同次运行中追加"""
    global _performance_file_initialized
    
    stats = get_performance_stats()
    if not stats:
        print("No performance data available.")
        return
    
    # 决定写入模式：本次运行第一次调用清空，后续调用追加
    mode = "w" if _performance_file_initialized else "a"
    
    with open("performance_summary.txt", mode, encoding="utf-8") as f:
        if not _performance_file_initialized:
            # 本次运行第一次调用，写入文件头
            f.write("="*80 + "\n")
            f.write("PERFORMANCE SUMMARY (sorted by total execution time)\n")
            f.write("="*80 + "\n")
            _performance_file_initialized = True
        else:
            # 本次运行后续调用，添加分隔符
            f.write("\n" + "-"*80 + "\n")
            f.write("ADDITIONAL PERFORMANCE DATA\n")
            f.write("-"*80 + "\n")
        
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time_ms'], reverse=True)
        
        for func_name, data in sorted_stats:
            f.write(f"{func_name:.<40} calls: {data['calls']:6d} | "
                   f"avg: {data['avg_time_ms']:8.3f}ms | "
                   f"total: {data['total_time_ms']:10.3f}ms\n")
        
        f.write("="*80 + "\n")
    
    status = "written to" if mode == "w" else "appended to"
    print(f"Performance summary has been {status} 'performance_summary.txt'")

def reset_performance_tracking(performance_file_initialized=True, hard=False):
    """重置性能跟踪状态，用于新的运行"""
    global _performance_file_initialized
    _performance_file_initialized = performance_file_initialized
    if hard:
        # 完全清零所有计数（回到程序启动状态）
        for wrapper in _profiled_functions.values():
            wrapper.call_count = 0
            wrapper.total_time = 0.0
            wrapper._baseline_calls = 0
            wrapper._baseline_total = 0.0

