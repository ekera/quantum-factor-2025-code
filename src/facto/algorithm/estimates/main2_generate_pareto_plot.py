from __future__ import annotations

import argparse
import math
import pathlib

from matplotlib import pyplot as plt

from facto.algorithm.estimates.main1_generate_cost_table_data import read_buckets


def save_cost_figure(*, csv_path: str | pathlib.Path, out_path: str | pathlib.Path | None, title: str | None = None, show: bool = False):
    buckets = read_buckets(csv_path)

    # Comparisons with other estimates.
    ge21_costs = None;
    cfs24_costs = None;

    def ge21_cost(n, n_e):
        return ([0.2 * n_e * (n ** 2) + 0.0003 * n_e * (n ** 2) * math.log2(n)],
                [3 * n + math.ceil(0.002 * n * math.log2(n))]);

    def ge21_cost_rsa(n):
        m = n / 2 - 1;
        l = m - 20;
        return ge21_cost(n, n_e = m + 2 * l);

    nist_model = {
      512:   56,
      768:   72,
      1024:  80,
      2048: 112,
      3072: 128,
      4096: 152,
      6144: 176,
      8192: 200
    }

    if title == "RSA IFP": # Note: Temporary solution to use the title.
        ge21_costs = {
            n: ge21_cost_rsa(n) for n in [1024, 2048, 3072, 4096]
        }

        cfs24_costs = {
            2048: ([2**40.87], [1730]),
            3072: ([2**42.70], [2415]),
            4096: ([2**43.49], [3096]),
            6144: ([2**45.36], [4430]),
            8192: ([2**46.63], [5781]),
        }
    elif title == "Short DLP in safe-prime groups":
        ge21_costs = {
            n: ge21_cost(n, 3 * 2 * nist_model[n]) for n in [1024, 2048, 3072, 4096]
        }

        cfs24_costs = {
            2048: ([2**36.47], [684]),
            3072: ([2**37.42], [719]),
            4096: ([2**38.30], [774]),
            6144: ([2**39.28], [831]),
            8192: ([2**40.05], [885]),
        }
    elif title == "DLP in Schnorr groups of known order":
        ge21_costs = {
            n: ge21_cost(n, 2 * (2 * nist_model[n] + 5)) for n in [1024, 2048, 3072, 4096]
        }

    # Plot.
    k = 0
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    for n, v in sorted(buckets.items()):
        xs = []
        ys = []
        best_tofs = float('inf')
        for qubits, result in sorted(v.items()):
            tofs = result["tofs"]
            n = result["modulus_bitlength"]
            if tofs * 1.05 > best_tofs:
                continue
            best_tofs = tofs
            xs.append(tofs)
            ys.append(qubits)

        # Clip terrible qubit tradeoffs as Toffolis skyrocket at end of curve.
        ys = ys[::-1]
        xs = xs[::-1]
        while len(ys) > 2 and 0.997 < abs(ys[-1] / ys[-2]) < 1.003:
            xs.pop()
            ys.pop()
        ys = ys[::-1]
        xs = xs[::-1]

        ax.text(xs[0], ys[0], f'  $n={n}$', horizontalalignment='left', verticalalignment='top', fontsize=8)
        ax.plot(xs, ys, label=f'$n={n}$', zorder=100, marker='.', color=f'C{k}')

        if None != ge21_costs:
          if n in ge21_costs:
              xs2, ys2 = ge21_costs[n]
              ax.plot(xs2, ys2, zorder=100, marker='s', color=f'C{k}')
              ax.text(xs2[0], ys2[0], f'  [GE21]\n  $n={n}$', horizontalalignment='left', verticalalignment='top', fontsize=8)

        if None != cfs24_costs:
          if n in cfs24_costs:
              xs2, ys2 = cfs24_costs[n]
              ax.plot(xs2, ys2, zorder=100, marker='p', color=f'C{k}')
              ax.text(xs2[0], ys2[0], f'  [CFS24]\n  $n={n}$', horizontalalignment='left', verticalalignment='top', fontsize=8)

        k += 1

    ax.loglog()

    ax.set_ylim(2 * 10**2, 1.5 * 10**4)
    ax.set_xlim(10**7, 3 * 10**14)

    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray')

    if None != title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel(f'Toffoli gates (overall summed over all runs required)', fontsize=13)
    ax.set_ylabel(fr'Logical qubits', fontsize=13)
    ax.legend(loc='lower right')
    fig.set_dpi(200)
    fig.set_size_inches(15, 7)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')
    if show or out_path is None:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    if args.out_path is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
    save_cost_figure(
        csv_path=csv_path,
        out_path=out_path,
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
