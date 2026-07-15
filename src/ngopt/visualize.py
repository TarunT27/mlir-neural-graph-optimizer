from __future__ import annotations

from html import escape

from ngopt.ir import Graph

_COLORS = {
    "ng.constant": ("#1e293b", "#94a3b8"),
    "ng.add": ("#0f766e", "#5eead4"),
    "ng.mul": ("#1d4ed8", "#93c5fd"),
    "ng.matmul": ("#6d28d9", "#c4b5fd"),
    "ng.relu": ("#be123c", "#fda4af"),
    "ng.linear_relu": ("#c2410c", "#fdba74"),
}


def graph_to_svg(graph: Graph) -> str:
    """Create a deterministic, dependency-free graph visualization."""
    nodes = [
        *(value.name for value in graph.inputs),
        *(operation.result for operation in graph.operations),
    ]
    positions = {name: (130 + index * 190, 130) for index, name in enumerate(nodes)}
    width = max(420, 260 + max(0, len(nodes) - 1) * 190)
    edges: list[str] = []
    for operation in graph.operations:
        target_x, target_y = positions[operation.result]
        for operand in operation.operands:
            if operand not in positions:
                continue
            source_x, source_y = positions[operand]
            edges.append(
                f'<path d="M {source_x + 62} {source_y} C {source_x + 95} {source_y}, '
                f'{target_x - 95} {target_y}, {target_x - 62} {target_y}" '
                'fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    node_markup: list[str] = []
    input_names = {value.name for value in graph.inputs}
    operation_by_result = {operation.result: operation for operation in graph.operations}
    for name in nodes:
        x, y = positions[name]
        if name in input_names:
            label = "input"
            fill, stroke = "#082f49", "#38bdf8"
        else:
            label = operation_by_result[name].op.removeprefix("ng.")
            fill, stroke = _COLORS[operation_by_result[name].op]
        node_markup.extend(
            [
                f'<rect x="{x - 62}" y="{y - 32}" width="124" height="64" rx="14" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                f'<text x="{x}" y="{y - 4}" text-anchor="middle" fill="#f8fafc" '
                'font-family="Segoe UI, sans-serif" font-size="15" font-weight="700">'
                f"{escape(label)}</text>",
                f'<text x="{x}" y="{y + 17}" text-anchor="middle" fill="#cbd5e1" '
                f'font-family="ui-monospace, monospace" font-size="11">{escape(name)}</text>',
            ]
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="260" '
            f'viewBox="0 0 {width} 260">',
            "  <defs>",
            '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
            '<stop stop-color="#020617"/><stop offset="1" stop-color="#172554"/>'
            "</linearGradient>",
            '    <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" '
            'refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" '
            'fill="#64748b"/></marker>',
            "  </defs>",
            f'  <rect width="{width}" height="260" rx="22" fill="url(#bg)"/>',
            '  <text x="32" y="45" fill="#e2e8f0" '
            'font-family="Segoe UI, sans-serif" font-size="20" font-weight="700">'
            f"@{escape(graph.name)}</text>",
            *(f"  {edge}" for edge in edges),
            *(f"  {node}" for node in node_markup),
            "</svg>",
            "",
        ]
    )
