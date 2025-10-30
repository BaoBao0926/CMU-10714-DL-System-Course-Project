

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_trace_grouped(trace_log):
    """print treee trace"""
    children_map = {}
    for entry in trace_log:
        parent = entry.get("parent")
        children_map.setdefault(parent, []).append(entry)

    def _group_by_prefix(nodes):
        grouped = {}
        for node in nodes:
            prefix = node["name"].split(".")[0]  # e.g., net1 from net1.0
            grouped.setdefault(prefix, []).append(node)
        return grouped

    def _print_subtree(parent=None, indent=0):
        nodes = children_map.get(parent, [])
        grouped = _group_by_prefix(nodes)
        pad = "  " * indent

        for group, members in grouped.items():
            if len(members) > 1:
                print(f"{pad}• {group} (Sequential)")
                for node in members:
                    print(f"{pad}  ├─ {node['name']} → {node['module_type']}")
            else:
                node = members[0]
                name = node["name"]
                op = node["op"]
                module = node["module_type"]
                needle = node["needle_type"]
                note = f"  # {node['note']}" if node["note"] else ""
                # print(f"{pad}• {name} ({op}) → {module} → {needle}{note}")
                print(f"{pad}• {name} ({op}) → {needle}{note}")

            # print sub_node
            for node in members:
                _print_subtree(node["name"], indent + 2)

    print("========Grouped Tree View ========")
    _print_subtree(None)
    print("========================================")

