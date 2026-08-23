"""Lightweight Mermaid-based graph visualization for Canonical IR and GGML Dialect."""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.tensor import StorageClass


def _escape(text: str) -> str:
    """Escapes labels for Mermaid syntax."""
    return text.replace('"', "'").replace("\n", " ")


def _get_node_category(opcode_name: str) -> str:
    """Categorizes opcode for Mermaid visual styling."""
    name = opcode_name.upper()
    if any(k in name for k in ("MATMUL", "LINEAR", "CONV", "MUL_MAT")):
        return "compute"
    if any(k in name for k in ("RELU", "GELU", "SILU", "SIGMOID", "TANH", "SOFTMAX")):
        return "activation"
    if any(k in name for k in ("NORM", "SWIGLU", "BIAS_GELU", "ROPE", "FUSED")):
        return "norm_fused"
    if any(
        k in name for k in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "SLICE", "CONCAT", "CONT")
    ):
        return "memory"
    return "elementwise"


def graph_to_mermaid(
    graph: Graph | GGMLExecutionGraph,
    title: str | None = None,
    direction: str = "TD",
) -> str:
    """Generates Mermaid flowchart diagram markdown for a Canonical IR or GGML Dialect graph.

    Args:
        graph: Canonical IR Graph or GGML Dialect GGMLExecutionGraph.
        title: Optional diagram title.
        direction: Flowchart direction ('TD' for top-down, 'LR' for left-to-right).

    Returns:
        String containing valid Mermaid flowchart code.

    Example:
        >>> mmd = graph_to_mermaid(graph, title="MiniLM Canonical IR")
        >>> print(mmd)
    """
    lines: list[str] = [f"graph {direction}"]

    graph_name = title or getattr(graph, "name", "Graph")
    lines.append(f'    subgraph SG_{_escape(graph_name)}["{_escape(graph_name)}"]')

    # Producer map: tensor_id -> producer node_id or input label
    producer_node: dict[int, str] = {}
    is_ggml = isinstance(graph, GGMLExecutionGraph)

    # 1. Inputs subgraph / nodes
    input_ids = list(
        dict.fromkeys(
            list(graph.inputs)
            + [
                tid
                for tid, t in graph.tensors.items()
                if getattr(t, "storage", None) == StorageClass.INPUT
            ]
        )
    )
    for inp_id in input_ids:
        t = graph.tensors.get(inp_id)
        if t:
            tname = _escape(t.name or f"in_{inp_id}")
            shape_str = f"ne: {t.ne}" if is_ggml else str(t.shape)
            node_key = f"INP_{inp_id}"
            lines.append(
                f'        {node_key}(["Input: {tname}<br/><b>{shape_str}</b>"]):::inputNode'
            )
            producer_node[inp_id] = node_key

    # 2. Parameter tensors
    param_ids = list(
        dict.fromkeys(
            list(graph.parameters)
            + [
                tid
                for tid, t in graph.tensors.items()
                if getattr(t, "storage", None) in (StorageClass.PARAMETER, StorageClass.CONSTANT)
            ]
        )
    )
    for param_id in param_ids:
        t = graph.tensors.get(param_id)
        if t:
            pname = _escape(t.name or f"param_{param_id}")
            shape_str = f"ne: {t.ne}" if is_ggml else str(t.shape)
            node_key = f"PARAM_{param_id}"
            lines.append(f'        {node_key}[("{pname}<br/><i>{shape_str}</i>")]:::paramNode')
            producer_node[param_id] = node_key

    # 3. Operations
    nodes = graph.nodes
    for node in nodes:
        op_id = f"OP_{node.id}"
        op_label: str
        cat: str
        if is_ggml:
            op_enum = GGMLOpCode(node.opcode) if isinstance(node.opcode, int) else node.opcode
            op_name = op_enum.name if hasattr(op_enum, "name") else str(node.opcode)
            cat = _get_node_category(op_name)
            op_label = f"#{node.id}: {op_name}"
            if node.name:
                op_label += f"<br/><i>{_escape(node.name)}</i>"
        else:
            op_enum = OpCode(node.opcode) if isinstance(node.opcode, (str, int)) else node.opcode
            op_name = op_enum.value if hasattr(op_enum, "value") else str(node.opcode)
            cat = _get_node_category(op_name)
            op_label = f"#{node.id}: {op_name.upper()}"
            if node.name:
                op_label += f"<br/><i>{_escape(node.name)}</i>"

        lines.append(f'        {op_id}["{op_label}"]:::{cat}')

        for out_id in node.outputs:
            producer_node[out_id] = op_id

    # 4. Edges
    for node in nodes:
        op_id = f"OP_{node.id}"
        for in_id in node.inputs:
            src = producer_node.get(in_id)
            if src:
                t = graph.tensors.get(in_id)
                edge_label = ""
                if t:
                    shape_repr = f"{t.ne}" if is_ggml else f"{t.shape}"
                    edge_label = f'|"{shape_repr}"|'
                lines.append(f"        {src} -->{edge_label} {op_id}")

    # 5. Output sinks
    for out_id in graph.outputs:
        src = producer_node.get(out_id)
        if src:
            t = graph.tensors.get(out_id)
            shape_str = f"ne: {t.ne}" if (is_ggml and t) else (str(t.shape) if t else "")
            out_node = f"OUT_{out_id}"
            lines.append(
                f'        {out_node}(["Output: #{out_id}<br/><b>{shape_str}</b>"]):::outputNode'
            )
            lines.append(f"        {src} --> {out_node}")

    lines.append("    end")

    # 6. Styling definitions
    lines.append("")
    lines.append("    classDef compute fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;")
    lines.append(
        "    classDef activation fill:#e8f8f5,stroke:#26a69a,stroke-width:2px,color:#004d40;"
    )
    lines.append(
        "    classDef norm_fused fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;"
    )
    lines.append("    classDef memory fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;")
    lines.append(
        "    classDef elementwise fill:#f9fbe7,stroke:#9e9d24,stroke-width:2px,color:#33691e;"
    )
    lines.append(
        "    classDef inputNode fill:#e0f2f1,stroke:#00897b,stroke-width:2px,stroke-dasharray: 5 5,color:#004d40;"
    )
    lines.append(
        "    classDef paramNode fill:#eceff1,stroke:#78909c,stroke-width:1px,color:#37474f;"
    )
    lines.append(
        "    classDef outputNode fill:#fce4ec,stroke:#d81b60,stroke-width:2px,color:#880e4f;"
    )

    return "\n".join(lines)


def visualize(
    graph: Graph | GGMLExecutionGraph,
    output_path: str | Path = "graph.html",
    format: str = "auto",
    title: str | None = None,
) -> Path:
    """Visualizes an IR graph and exports it to HTML, Mermaid markdown (.mmd), SVG, or PNG.

    Args:
        graph: Canonical IR Graph or GGML Dialect execution graph.
        output_path: Target file path (e.g. 'model.html', 'model.mmd', 'model.svg', 'model.png').
        format: Target format ('auto', 'html', 'mmd', 'svg', 'png').
        title: Optional title for the diagram.

    Returns:
        Path to the generated visualization file.
    """
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    chosen_format = format.lower()
    if chosen_format == "auto":
        ext = p.suffix.lower().lstrip(".")
        chosen_format = ext if ext in ("html", "mmd", "svg", "png") else "html"

    mmd_code = graph_to_mermaid(graph, title=title)

    if chosen_format == "mmd":
        p.write_text(mmd_code, encoding="utf-8")
        return p

    if chosen_format == "html":
        escaped_mmd = html.escape(mmd_code)
        graph_title = title or getattr(graph, "name", "ggmlc Computation Graph")
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(graph_title)} - ggmlc</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8fafc;
            color: #0f172a;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0;
            font-size: 1.5rem;
            color: #1e293b;
        }}
        .container {{
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            padding: 24px;
            overflow: auto;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html.escape(graph_title)}</h1>
        <span style="color: #64748b; font-size: 0.9rem;">Generated by ggmlc</span>
    </div>
    <div class="container">
        <pre class="mermaid">
{escaped_mmd}
        </pre>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: false, htmlLabels: true, curve: 'basis' }}
        }});
    </script>
</body>
</html>
"""
        p.write_text(html_content, encoding="utf-8")
        return p

    if chosen_format in ("png", "svg", "pdf"):
        # 1. Try pure-Python mermaidx / mmdc engine (QuickJS + resvg, zero Node.js needed)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                import mermaidx as mmdc_pkg  # type: ignore
            except ImportError:
                try:
                    import mmdc as mmdc_pkg  # type: ignore
                except ImportError:
                    mmdc_pkg = None

        if mmdc_pkg is not None:
            try:
                diagram = mmdc_pkg.render(mmd_code)
                diagram.save(str(p), format=chosen_format)
                return p
            except (RuntimeError, ValueError, OSError):
                pass

        # 2. Check if external mmdc CLI binary is available
        mmdc_bin = shutil.which("mmdc")
        if mmdc_bin:
            tmp_mmd = p.with_suffix(".mmd")
            tmp_mmd.write_text(mmd_code, encoding="utf-8")
            try:
                subprocess.run([mmdc_bin, "-i", str(tmp_mmd), "-o", str(p)], check=True)
                if tmp_mmd.exists() and tmp_mmd != p:
                    tmp_mmd.unlink()
                return p
            except (subprocess.SubprocessError, OSError):
                pass

        # 3. Fallback to HTML if neither Python package nor CLI is available
        html_path = p.with_suffix(".html")
        return visualize(graph, output_path=html_path, format="html", title=title)

    p.write_text(mmd_code, encoding="utf-8")
    return p
