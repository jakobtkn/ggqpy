rule all:
    input:
        "output/triangle-test.1.4.tex",
        "output/triangle-test.16.4.tex",
        "output/triangle-test.16.8.tex",
        # expand("quads/nystrom.{number_parameters}.{order}.quad", number_parameters=COUNTS, order=ORDERS)

FILES = ["triangle-test.16.4.tex"]
rule tex:
    input:
        expand("../report/output/{file}", file=FILES)

rule copy_file_to_tex:
    input: "output/{file}"
    output: "../report/output/{file}"
    shell:
        "cp output/{wildcards.file} ../report/output/{wildcards.file}"

rule generate_quadrature:
    input:
    output:
        "quads/nystrom.{number_parameters}.{order}.quad"
    shell:
        "python3 examples/generate_quad.py {wildcards.number_parameters} {wildcards.order} quads/nystrom.{wildcards.number_parameters}.{wildcards.order}.quad"

ALPHAS = [0.5,0.1,1e-3,1e-9]
rule generate_table:
    input:
        "quads/nystrom.{number_parameters}.{order}.quad"
    output:
        "output/triangle-test.{number_parameters}.{order}.tex"
    shell:
        "python3 examples/triangle-test.py {wildcards.number_parameters} {wildcards.order} > output/triangle-test.{wildcards.number_parameters}.{wildcards.order}.tex"

