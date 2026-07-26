#!/usr/bin/env python3
from pathlib import Path
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
bootstrap()
from scripts import run_xtb_singlepoint_lsgo as implementation
def main():
    implementation.OUT=Path(__file__).resolve().parents[1]/'reports/ecir_mvr/lsgo_bond_confirm';result=implementation.main()
    if result==0:print('BOND_CONFIRM_XTB_COMPLETED')
    return result
if __name__=='__main__':raise SystemExit(main())
