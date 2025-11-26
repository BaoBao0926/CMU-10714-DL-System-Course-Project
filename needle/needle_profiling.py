import time
import functools
from collections import defaultdict

# 全局注册表存储所有被装饰的函数
_profiled_functions = {}

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
    """打印性能摘要，按总执行时间排序"""
    stats = get_performance_stats()
    if not stats:
        print("No performance data available.")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (sorted by total execution time)")
    print("="*80)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time_ms'], reverse=True)
    
    for func_name, data in sorted_stats:
        print(f"{func_name:.<30} calls: {data['calls']:6d} | "
              f"avg: {data['avg_time_ms']:8.3f}ms | "
              f"total: {data['total_time_ms']:10.3f}ms")
    
    print("="*80)

