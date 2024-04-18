rule all:
    input:
        "output/experiment_triangle.1.4.tex",
        "output/experiment_triangle.16.4.tex",
        "output/experiment_triangle.16.8.tex",
        # expand("quads/nystrom.{number_parameters}.{order}.quad", number_parameters=COUNTS, order=ORDERS)



R = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.4,0.7,0.8,1.0]
THETA = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.25,0.5,1.0,1.5,3.0, 3.14, 3.14159265359]

rule make_config:
    input:
    output:
        "quads/nystrom/breakpoints_r",
        "quads/nystrom/breakpoints_theta"
    shell:
        "echo {R} > quads/nystrom/breakpoints_r | echo {THETA} > quads/nystrom/breakpoints_theta"

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
        "output/experiment_triangle.{number_parameters}.{order}.tex"
    shell:
        "python3 examples/experiment_triangle.py {wildcards.number_parameters} {wildcards.order} > output/experiment_triangle.{wildcards.number_parameters}.{wildcards.order}.tex"






FILES = ["experiment_triangle.16.4.tex"]
rule tex:
    input:
        expand("../report/output/{file}", file=FILES)

rule copy_file_to_tex:
    input: "output/{file}"
    output: "../report/output/{file}"
    shell:
        "cp output/{wildcards.file} ../report/output/{wildcards.file}"