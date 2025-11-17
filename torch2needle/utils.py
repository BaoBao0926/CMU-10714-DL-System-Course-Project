

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_trace_grouped(trace_log):
    """
    按顺序打印转换追踪，展示每一层的转换过程
    """
    print("\n" + "="*100)
    print("📊 TORCH → NEEDLE 转换追踪 (按执行顺序)")
    print("="*100)
    print(f"{'序号':<6} {'节点名称':<20} {'操作类型':<15} {'PyTorch类型':<20} {'→':<3} {'Needle类型':<20} {'备注'}")
    print("-"*100)
    
    for idx, entry in enumerate(trace_log, 1):
        name = entry.get('name', '')
        op = entry.get('op', '')
        torch_type = entry.get('module_type', '') or ''
        needle_type = entry.get('needle_type', '') or ''
        note = entry.get('note', '')
        
        # 根据操作类型添加符号
        if op == 'placeholder':
            symbol = "🔵"
        elif op == 'call_module':
            symbol = "🟢"
        elif op == 'call_function':
            symbol = "🟡"
        elif op == 'output':
            symbol = "🔴"
        else:
            symbol = "⚪"
        
        # 打印每一行
        print(f"{idx:<6} {symbol} {name:<18} {op:<15} {torch_type:<20} → {needle_type:<20} {note}")
    
    print("="*100)
    
    # 统计信息
    total = len(trace_log)
    modules = sum(1 for e in trace_log if e['op'] == 'call_module')
    functions = sum(1 for e in trace_log if e['op'] == 'call_function')
    
    print(f"\n📈 统计: 总共 {total} 个节点 | {modules} 个模块 | {functions} 个函数操作")
    print(f"图例: 🔵 输入 | 🟢 模块 | 🟡 函数 | 🔴 输出")
    print("="*100 + "\n")

