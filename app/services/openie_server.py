from openie import StanfordOpenIE

# Optional: Use any properties
properties = {
    'openie.affinity_probability_cap': 1 / 3,
    'openie.resolve_coref': True,
}

# Start OpenIE Server only once
openie_client = StanfordOpenIE(properties=properties)