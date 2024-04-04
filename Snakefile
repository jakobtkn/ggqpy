COUNTS = [32]
ORDERS = [4,8]

rule all:
    input:
        "output/triangle-test.tex",
        # expand("quads/nystrom.{number_parameters}.{order}.quad", number_parameters=COUNTS, order=ORDERS)

FILES = ["triangle-test.tex"]
rule tex:
    input:
        expand("../report/output/{file}", file=FILES)

rule copy_file_to_tex:
    input: "output/{file}"
    output: "../report/output/{file}"
    shell:
        "cp output/{input.file} ../report/output/{input.file}"

rule generate_quadrature:
    input:
    output:
        "quads/nystrom.{number_parameters}.{order}.quad"
    shell:
        "python3 examples/generate_quad.py {wildcards.number_parameters} {wildcards.order} quads/nystrom.{wildcards.number_parameters}.{wildcards.order}.quad"

ALPHAS = [0.5,0.1,1e-3,1e-9]
rule generate_table:
    input:
        "quads/nystrom.16.4.quad"
    output:
        "output/triangle-test.tex"
    shell:
        "python3 examples/triangle-test.py 16 4 > output/triangle-test.tex"

