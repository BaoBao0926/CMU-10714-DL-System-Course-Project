

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_trace_grouped(trace_log):
    """
    Print conversion trace in order, showing the conversion process of each layer
    """
    print("\n" + "="*100)
    print("📊 TORCH → NEEDLE Conversion Trace (Execution Order)")
    print("="*100)
    print(f"{'Index':<6} {'Node Name':<20} {'Op Type':<15} {'PyTorch Type':<20} {'→':<3} {'Needle Type':<20} {'Note'}")
    print("-"*100)
    
    for idx, entry in enumerate(trace_log, 1):
        name = entry.get('name', '')
        op = entry.get('op', '')
        torch_type = entry.get('module_type', '') or ''
        needle_type = entry.get('needle_type', '') or ''
        note = entry.get('note', '')
        
        # Add symbol based on operation type
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
        
        # Print each line
        print(f"{idx:<6} {symbol} {name:<18} {op:<15} {torch_type:<20} → {needle_type:<20} {note}")
    
    print("="*100)
    
    # Statistics
    total = len(trace_log)
    modules = sum(1 for e in trace_log if e['op'] == 'call_module')
    functions = sum(1 for e in trace_log if e['op'] == 'call_function')
    
    print(f"\n📈 Stats: Total {total} nodes | {modules} modules | {functions} function ops")
    print(f"Legend: 🔵 Input | 🟢 Module | 🟡 Function | 🔴 Output")
    print("="*100 + "\n")

