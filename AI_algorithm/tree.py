import graphviz
# import copy # copy 模块不再需要
import os


class Node:
    """定义树的节点"""
    _id_counter = 0

    def __init__(self, value, depth, is_extension=False):
        self.value = value
        self.depth = depth
        self.children = []
        self.id = Node._id_counter
        self.is_extension = is_extension
        Node._id_counter += 1

    def __repr__(self):
        return f"Node(val={self.value}, d={self.depth})"


# ===================================================================
# 新增的树复制函数
# ===================================================================
def copy_tree(node: Node) -> Node:
    """
    递归地手动复制一棵树，以避免 deepcopy 的副作用。
    """
    # 1. 创建当前节点的新副本。
    # 这会调用 Node.__init__ 并分配一个新的唯一 ID。
    new_node = Node(node.value, node.depth, node.is_extension)

    # 2. 递归地复制所有子节点，并将它们附加到新副本的 children 列表中。
    for child in node.children:
        new_node.children.append(copy_tree(child))

    return new_node


# ===================================================================

def _build_recursive(parent_node: Node, k: int, n: int):
    """递归构建树的辅助函数 (无变化)"""
    if parent_node.depth == k:
        return

    parent_val = parent_node.value
    parent_depth = parent_node.depth
    min_child_val = parent_val + 1
    max_child_val = n - (k - parent_depth - 1)

    for child_val in range(min_child_val, max_child_val + 1):
        child_node = Node(child_val, parent_depth + 1)
        parent_node.children.append(child_node)
        _build_recursive(child_node, k, n)


def generate_tree(k: int, n: int) -> Node | None:
    """生成满足条件的原始组合树 (无变化)"""
    if n < k:
        print(f"无法构建树：n ({n}) 必须大于等于 k ({k})")
        return None
    Node._id_counter = 0
    root = Node(value=0, depth=0)
    _build_recursive(root, k, n)
    return root


def _extend_leaves_recursive(node: Node, path_values: list, k: int, n: int):
    """递归辅助函数：找到叶子节点并附加剩余元素链表 (无变化)"""
    if node.depth == k:
        used_nums = set(path_values[1:])
        all_nums = set(range(1, n + 1))
        remaining_nums = sorted(list(all_nums - used_nums))
        current_node = node
        for val in remaining_nums:
            new_depth = current_node.depth + 1
            extension_node = Node(val, new_depth, is_extension=True)
            current_node.children.append(extension_node)
            current_node = extension_node
        return

    for child in node.children:
        _extend_leaves_recursive(child, path_values + [child.value], k, n)


def extend_tree(original_root: Node, k: int, n: int) -> Node | None:
    """
    在原始树的副本上，为每个叶子节点附加剩余元素的升序链表。
    """
    if not original_root:
        return None

    # ===================================================================
    # 这是关键的修改点：用我们自己的复制函数替换 deepcopy
    # extended_root = copy.deepcopy(original_root)  <-- 旧的、有问题的方法
    extended_root = copy_tree(original_root)  # <-- 新的、正确的方法
    # ===================================================================

    _extend_leaves_recursive(extended_root, [extended_root.value], k, n)
    return extended_root


def visualize_two_trees(original_root: Node, extended_root: Node, filename="comparison"):
    """
    使用 Graphviz 将两棵树并排可视化并保存到同一个文件中。(无变化)
    """
    if not original_root or not extended_root:
        print("至少有一棵树为空，无法可视化。")
        return

    dot = graphviz.Digraph(comment='Tree Comparison')
    dot.attr(rankdir='TB', splines='spline')

    def _add_tree_to_graph(graph, root_node):
        queue = [root_node]
        visited_ids = set()  # 防止意外的循环
        while queue:
            node = queue.pop(0)
            if node.id in visited_ids:
                continue
            visited_ids.add(node.id)

            color = 'lightcoral' if node.is_extension else 'lightblue'
            graph.node(str(node.id), str(node.value), shape='circle', style='filled', fillcolor=color)

            for child in node.children:
                graph.edge(str(node.id), str(child.id))
                queue.append(child)

    with dot.subgraph(name='cluster_0') as c:
        c.attr(label=f'Original Tree C({N}, {K})')
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        _add_tree_to_graph(c, original_root)

    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Extended Tree')
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        _add_tree_to_graph(c, extended_root)

    output_path = os.path.join("graph", filename)
    try:
        os.makedirs("graph", exist_ok=True)
        dot.render(output_path, format='png', view=True)
        print(f"\n图表已成功生成并保存为 '{output_path}.png'")
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n[错误] Graphviz 未找到！")
        print("请确保已经安装了 Graphviz 软件，并将其 bin 目录添加到了系统 PATH 环境变量中。")
    except Exception as e:
        print(f"\n生成图表时发生错误: {e}")


# --- 主程序 (无变化) ---
if __name__ == "__main__":
    K = 3
    N = 5

    print(f"--- K={K}, N={N} ---")

    print("1. 正在生成原始树...")
    original_tree = generate_tree(k=K, n=N)

    if original_tree:
        print("2. 正在生成扩展树...")
        extended_tree = extend_tree(original_tree, k=K, n=N)

        print("3. 正在可视化两棵树...")
        visualize_two_trees(
            original_tree,
            extended_tree,
            filename=f"comparison_k{K}_n{N}"
        )