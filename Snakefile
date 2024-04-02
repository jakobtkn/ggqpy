COUNTS = [2,4,8,16,32,64,128]
ORDERS = [4,6,8]

rule all:
    input:
        expand("quads/nystrom.{number_parameters}.{order}.quad", number_parameters=COUNTS, order=ORDERS)

rule generate_quadrature:
    input:
    output:
        "quads/nystrom.{number_parameters}.{order}.quad"
    shell:
        "python3 examples/generate_quad.py {wildcards.number_parameters} {wildcards.order} quads/nystrom.{wildcards.number_parameters}.{wildcards.order}.quad"
