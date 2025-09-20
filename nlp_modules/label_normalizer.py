def normalize_label(label: str) -> str:
    syn = {
        "cap_on_liability": "limitation_of_liability",
        "limitation_of_liabilities": "limitation_of_liability",
        "attorneys_fees": "attorney_fees",
        "attorney_fee": "attorney_fees",
        "fees_and_expenses": "attorney_fees",
        "security_breach": "data_breach",
        "data_security_breach": "data_breach",
        "renewal": "renewal_term",
        "auto_renewal": "renewal_term",
        "post_termination": "post_termination_services",
        "post_termination_obligations": "post_termination_services",
        "termination": "termination_obligations",
        "termination_rights": "termination_obligations",
        "termination_without_cause": "termination_for_convenience",
        "material_breach_termination": "termination_for_material_breach",
    }
    norm = label.lower().replace(" ", "_")
    return syn.get(norm, norm)
