@echo off
REM ============================================================================
REM  reproduce.bat -- rebuild every number, table and figure in the manuscript
REM  from the raw dataset and the split file, then verify that the results
REM  files still say what the manuscript says.
REM
REM  Needs:   conda env  cnt-vfm   (environment.yml or environment-cpu.yml)
REM           CNT_BASE   the data root: <CNT_BASE>\NIOSH Dataset\CNT-*\*.tif and
REM                      <CNT_BASE>\NIOSH Dataset\Masks\masks\*_mask.png
REM                      (see classification\REPRODUCE.md, "The data")
REM  Optional: CNT_TEX   directory holding main.tex / supplementary.tex.  Without
REM                      it step 7 checks against classification\manuscript_numbers.json,
REM                      the snapshot of the numbers as printed in the paper.
REM
REM  Stages 1-3 are the expensive ones (~3 h on an RTX 5080, mostly stage 3);
REM  they are skipped automatically when their outputs exist.  The shipped
REM  results\ directory therefore lets a first run finish in about ten minutes
REM  (feature caches + epoch sweep + figures + tables + verification).
REM  Pass  --force  to redo stage 3 from scratch (it renames the shipped
REM  results\analysis.json out of the way first).
REM ============================================================================
setlocal
set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"
set "CLS=%REPO%\classification"
set "SPLITS=%REPO%\splits\dataset_splits.pkl"
set "OUT=%REPO%\results"
set "PYTHONUNBUFFERED=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"

if "%CNT_BASE%"=="" (
  echo CNT_BASE is not set. Point it at the directory holding "NIOSH Dataset\" -- see classification\REPRODUCE.md.
  exit /b 1
)
if not exist "%CNT_BASE%\NIOSH Dataset" (
  echo "%CNT_BASE%\NIOSH Dataset" does not exist. See classification\REPRODUCE.md, "The data".
  exit /b 1
)
set "BENCHD=%CNT_BASE%\Encoder Benchmark"

REM Found and activated wherever conda is installed.
if not "%CONDA_DEFAULT_ENV%"=="cnt-vfm" (
  call "%REPO%\scripts\activate_env.bat" || exit /b 1
)

cd /d "%CLS%"

echo ==== 0. preflight =========================================================
python -u check_env.py || goto :fail
REM Figure 7 runs SAM ViT-B; fetched once (358 MB), skipped when present.
python -u "%REPO%\download_sam_weights.py" --model vit_b || goto :fail
python -u encoder_bench.py --stage features --group benchmark --splits "%SPLITS%" --out-dir "%BENCHD%" --dry-run || goto :fail
python -u check_cache_agreement.py --splits "%SPLITS%" || goto :fail

echo ==== 1-2. features (skipped per encoder when the cache exists) =============
python -u encoder_bench.py --stage features --group benchmark --splits "%SPLITS%" --out-dir "%BENCHD%" || goto :fail
python -u encoder_bench.py --stage features --encoders dinov2_b14 --target-grid 37 --taps 1,3,6,9,11 --splits "%SPLITS%" --out-dir "%BENCHD%" || goto :fail

echo ==== 3. probes + fine-tuned baselines + analysis ===========================
if "%~1"=="--force" if exist "%OUT%\analysis.json" (
  echo     --force: moving the shipped results aside to results\_previous\
  if not exist "%OUT%\_previous" mkdir "%OUT%\_previous"
  move /y "%OUT%\analysis.json" "%OUT%\_previous\" >nul
  move /y "%OUT%\results_probes.csv" "%OUT%\_previous\" >nul
  move /y "%OUT%\results_finetune.csv" "%OUT%\_previous\" >nul
  move /y "%OUT%\per_image_probe.json" "%OUT%\_previous\" >nul
  move /y "%OUT%\per_image_finetune.json" "%OUT%\_previous\" >nul
)
if exist "%OUT%\analysis.json" (
  echo     results\analysis.json exists -- skipping ^(pass --force to recompute; ~3 h^).
) else (
  python -u paper_results.py --splits "%SPLITS%" --out "%OUT%" --stages probes finetune analyse || goto :fail
)

echo ==== 4. epoch-budget sweep ================================================
python -u epoch_sweep.py "%SPLITS%" "%OUT%" || goto :fail

echo ==== 5. figures ===========================================================
python -u figures\make_confusion.py "%OUT%" "%OUT%\figures" || goto :fail
python -u figures\make_tsne.py "%SPLITS%" "%OUT%\figures" || goto :fail
python -u figures\make_sam_mosaic.py "%OUT%\figures" || goto :fail

echo ==== 6. tables ============================================================
python -u make_tables.py "%OUT%" || goto :fail

echo ==== 7. verify the manuscript numbers =====================================
if "%CNT_TEX%"=="" (
  python -u verify_manuscript.py "%OUT%" || goto :fail
) else (
  python -u verify_manuscript.py "%OUT%" --tex "%CNT_TEX%" || goto :fail
)

echo.
echo ============ REPRODUCED AND VERIFIED ============
goto :eof
:fail
echo.
echo ============ FAILED at the step above ============
exit /b 1
