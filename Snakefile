R = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.4,0.7,0.8,1.0]
THETA = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.25,0.5,1.0,1.5,3.0, 3.14, 3.14159265359]

nR = len(R)
nTHETA = len(THETA)

rule all:
    input:
        "output/experiment_triangle.4.tex",
        "output/experiment_triangle.8.tex",
        "output/experiment_triangle.16.tex",

rule make_config:
    input:
    output:
        "quads/nystrom.{order}/breakpoints_r",
        "quads/nystrom.{order}/breakpoints_theta"
    shell:
        "mkdir -p quads/nystrom.{wildcards.order} | echo {R} > quads/nystrom.{wildcards.order}/breakpoints_r | echo {THETA} > quads/nystrom.{wildcards.order}/breakpoints_theta"

rule generate_nystrom_quadrature:
    input:
        "quads/nystrom.{order}/breakpoints_r",
        "quads/nystrom.{order}/breakpoints_theta"
    output:
        "quads/nystrom.{order}/{r0_index}.{theta0_index}.quad"
    shell:
        "python3 examples/generate_nystrom_quad.py {wildcards.order} {wildcards.r0_index} {wildcards.theta0_index} quads/nystrom.{wildcards.order}/{wildcards.r0_index}.{wildcards.theta0_index}.quad"

ALPHAS = [0.5,0.1,1e-3,1e-9]
rule generate_table:
    input:
        "quads/nystrom.{order}/breakpoints_r",
        "quads/nystrom.{order}/breakpoints_theta",
        expand("quads/nystrom.{order}/{r0_index}.{theta0_index}.quad", r0_index=range(nR-1), theta0_index=range(nTHETA-1), allow_missing=True),
    output:
        "output/experiment_triangle.{order}.tex"
    shell:
        "python3 examples/experiment_triangle.py {wildcards.order} > output/experiment_triangle.{wildcards.order}.tex"


FILES = ["experiment_triangle.16.4.tex"]
rule tex:
    input:
        expand("../report/output/{file}", file=FILES)

rule copy_file_to_tex:
    input: "output/{file}"
    output: "../report/output/{file}"
    shell:
        "cp output/{wildcards.file} ../report/output/{wildcards.file}"
