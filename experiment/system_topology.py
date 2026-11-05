"""Physical unit operations, device placement and the notebook-compatible MSI grammar."""
from copy import deepcopy
import re
from urllib.parse import quote, unquote

PUMPS = {"masterflex", "ismatec"}
READ_ONLY = {"opcua", "mettler"}


def empty_topology():
    return {"version": 1, "modules": [], "unit_operations": []}


def token(prefix, identifier):
    # Escape delimiter characters even when urllib normally considers them safe.
    encoded = quote(str(identifier), safe="")
    for char, escaped in (("_", "%5F"), ("-", "%2D"), (".", "%2E"), ("~", "%7E")):
        encoded = encoded.replace(char, escaped)
    return prefix + encoded


def validate_topology(topology, configs, complete=True):
    if topology.get("version") != 1:
        raise ValueError("Unsupported system topology version")
    modules, units = {}, {}
    for row in topology.get("modules", []):
        code = row.get("code", "").strip()
        if not code or code in modules:
            raise ValueError("Modules need nonempty, unique identifier codes")
        modules[code] = row
    sensor_codes = set()
    for module, row in modules.items():
        for sensor in row.get("sensors", []):
            code = sensor.get("code", "").strip()
            if not code or code in sensor_codes or code in modules:
                raise ValueError("Module sensors need globally unique codes, different from module codes")
            if not sensor.get("name", "").strip():
                raise ValueError(f"Sensor {code}: enter a name")
            sensor_codes.add(code)
    for row in topology.get("unit_operations", []):
        code = row.get("code", "").strip()
        if not code or code in units:
            raise ValueError("Unit operations need nonempty, unique identifier codes")
        if row.get("module") not in modules:
            raise ValueError(f"Unit {code}: select an existing module")
        if not row.get("kind", "").strip() or not row.get("name", "").strip():
            raise ValueError(f"Unit {code}: enter a name and unit-operation type")
        units[code] = row
    if complete and not units:
        raise ValueError("Add at least one module and unit operation first")
    for device_id, config in configs.items():
        placement = config.get("connectivity", {})
        role = placement.get("role", "excluded")
        if role == "excluded":
            continue
        if role not in {"controlled", "read_only"}:
            raise ValueError(f"{device_id}: invalid system role")
        if not complete:
            continue  # Draft placements may be incomplete while editing units.
        module = placement.get("module")
        if module not in modules:
            raise ValueError(f"{device_id}: select a module")
        if config["type"] in PUMPS:
            if role != "controlled":
                raise ValueError(f"{device_id}: pumps use controlled-flow placement")
            source, target = placement.get("source"), placement.get("target")
            if source not in units or target not in units:
                raise ValueError(f"{device_id}: select source and target unit operations")
            # A pump is its own unit operation. Its owning module is independent
            # of the source/target vessels; either fluid link may cross modules.
        else:
            unit = placement.get("unit")
            if unit not in units:
                raise ValueError(f"{device_id}: select the unit operation it is connected to")
            if units[unit]["module"] != module:
                raise ValueError(f"{device_id}: device and connected unit must belong to the same module")
            if config["type"] in READ_ONLY and role != "read_only":
                raise ValueError(f"{device_id}: this device is a read-only probe")
    return modules, units


def encode_msi(topology, configs):
    modules, units = validate_topology(topology, configs)
    records = {token("U_", code): {"module": row["module"], "read": [], "control": [], "out": []}
               for code, row in units.items()}
    for device, config in sorted(configs.items()):
        p = config.get("connectivity", {})
        if p.get("role", "excluded") == "excluded":
            continue
        if config["type"] in PUMPS:
            pump = token("U_P_", device)
            records[pump] = {"module": p["module"], "read": [], "control": [token("C_F_", device)], "out": [token("U_", p["target"])]}
            records[token("U_", p["source"])]["out"].append(pump)
        elif config["type"] == "jkem":
            stirrer = token("U_S_", device)
            passive = p["role"] == "read_only"
            records[stirrer] = {"module": p["module"],
                "read": [token("R_TQ_", device)] + ([token("R_SR_", device)] if passive else []),
                "control": [] if passive else [token("C_SR_", device)],
                "out": [token("U_", p["unit"])]}
        else:
            field, prefix = ("read", "R_") if p["role"] == "read_only" else ("control", "C_")
            records[token("U_", p["unit"])][field].append(token(prefix, device))
    parts = []
    for module in sorted(modules):
        body, outputs = [], set()
        for code, record in sorted(records.items()):
            if record["module"] != module:
                continue
            body.append(f"{code}({','.join(sorted(record['read']))}){{{','.join(sorted(record['control']))}}}[{','.join(sorted(set(record['out'])))}]")
            outputs.update(records[target]["module"] for target in record["out"] if records[target]["module"] != module)
        sensors = ','.join(token('S_', s['code']) for s in sorted(modules[module].get('sensors', []), key=lambda s: s['code']))
        body_text = "__".join(body) + f"___{token('S_', module)}[[{sensors}]]"
        parts.append(f"{token('M_', module)}{{{{{body_text}}}}}(({','.join(token('M_', target) for target in sorted(outputs))}))")
    result = "--".join(parts)
    parse_msi(result)  # Validate the same grammar used by the graph decoder.
    return result


def parse_msi(code):
    """Parse the full grammar, including empty modules, cycles and multiple outputs."""
    modules, nodes, edges = {}, {}, []
    def items(text):
        return [item.strip() for item in text.split(",") if item.strip()]
    for chunk in code.strip().split("--"):
        match = re.fullmatch(r"(M_[^{}()]+)\{\{(.*?)\}\}\(\((.*?)\)\)", chunk.strip(), re.S)
        if not match:
            raise ValueError("Invalid MSI module syntax")
        module, body, outputs = match.groups()
        if module in modules:
            raise ValueError(f"Duplicate MSI module: {module}")
        modules[module] = {"id": module, "outputs": items(outputs)}
        if body.count("___") != 1:
            raise ValueError(f"{module}: expected one sensor-system separator ___")
        unit_text, sensors = body.split("___")
        sensor_match = re.fullmatch(r"(S_[^\[\]]+)\[\[([^\[\]]*)\]\]", sensors)
        if not sensor_match:
            raise ValueError(f"{module}: invalid sensor system")
        for entry in unit_text.split("__") if unit_text else []:
            unit_match = re.fullmatch(r"(U_[^(){}\[\],]+)\(([^()]*)\)\{([^{}]*)\}\[([^\[\]]*)\]", entry)
            if not unit_match:
                raise ValueError(f"{module}: invalid unit operation: {entry}")
            unit, read, controlled, targets = unit_match.groups()
            if unit in nodes:
                raise ValueError(f"Duplicate MSI identifier: {unit}")
            nodes[unit] = {"id": unit, "module": module, "role": "unit"}
            for field, role, prefix in ((read, "read_only", "R_"), (controlled, "controlled", "C_")):
                for device in items(field):
                    if not device.startswith(prefix) or device in nodes:
                        raise ValueError(f"Invalid or duplicated MSI device: {device}")
                    nodes[device] = {"id": device, "module": module, "role": role}
                    edges.append({"source": unit if role == "read_only" else device,
                                  "target": device if role == "read_only" else unit, "kind": role})
            # Stirrer-to-vessel links describe placement, rather than liquid flow.
            edges.extend({"source": unit, "target": target,
                          "kind": "attachment" if unit.startswith("U_S_") else "flow"} for target in items(targets))
        group, sensor_list = sensor_match.groups()
        if items(sensor_list):
            if group in nodes:
                raise ValueError(f"Duplicate sensor group: {group}")
            nodes[group] = {"id": group, "module": module, "role": "sensor_group"}
            for sensor in items(sensor_list):
                if not sensor.startswith("S_") or sensor in nodes:
                    raise ValueError(f"Invalid or duplicated sensor: {sensor}")
                nodes[sensor] = {"id": sensor, "module": module, "role": "sensor"}
                edges.append({"source": group, "target": sensor, "kind": "sensor"})
    for module in modules.values():
        if any(target not in modules for target in module["outputs"]):
            raise ValueError(f"Unknown target module in {module['id']}")
    for edge in edges:
        if edge["target"] not in nodes:
            raise ValueError(f"Unknown connected MSI node: {edge['target']}")
    return {"modules": list(modules.values()), "nodes": list(nodes.values()), "edges": edges}


def system_graph(topology, configs):
    code = encode_msi(topology, configs)
    graph = parse_msi(code)
    labels = {token("M_", m["code"]): m.get("name", m["code"]) for m in topology["modules"]}
    details = {token("U_", u["code"]): {"label": f"{u['name']}\n{u['code']} · {u['kind']}"} for u in topology["unit_operations"]}
    for module in topology['modules']:
        details[token('S_', module['code'])] = {'label': f"Module sensors: {module.get('name', module['code'])}"}
        for sensor in module.get('sensors', []):
            details[token('S_', sensor['code'])] = {'label': f"{sensor['name']}\n{sensor['code']}"}
    for device, config in configs.items():
        p = config.get("connectivity", {})
        if p.get("role", "excluded") == "excluded":
            continue
        if config["type"] in PUMPS:
            details[token("U_P_", device)] = {"label": f"Pump: {config['name']}\n{device}", "device_id": device, "equipment": "pump"}
            details[token("C_F_", device)] = {"label": f"Flow control\n{device}"}
        elif config["type"] == "jkem":
            details[token("U_S_", device)] = {"label": f"Stirrer: {config['name']}\n{device}", "device_id": device, "equipment": "stirrer"}
            details[token("R_TQ_", device)] = {"label": f"Read-only torque\n{device}", "device_id": device, "signal": "torque_ozin"}
            details[token("C_SR_", device)] = {"label": f"Stir-rate control (RPM)\n{device}", "device_id": device, "signal": "target"}
            details[token("R_SR_", device)] = {"label": f"Read-only stir rate (RPM)\n{device}", "device_id": device, "signal": "speed_rpm"}
        else:
            details[token("R_" if p["role"] == "read_only" else "C_", device)] = {"label": f"{config['name']}\n{device}"}
    for node in graph["nodes"]:
        node.update(details.get(node["id"], {}))
    for module in graph["modules"]:
        module["label"] = labels[module["id"]]
    return code, graph
