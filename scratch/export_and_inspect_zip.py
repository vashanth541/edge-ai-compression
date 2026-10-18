import sys, json, hashlib, zipfile, io
from pathlib import Path

PROJECT_ROOT = Path(r'C:\Users\ELCOT\OneDrive\Documents\edge-ai-compression')
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.dashboard.model_analyzer import analyze_tflite_bytes
from src.dashboard.compatibility_checker import check_esp32s3_compatibility, generate_compatibility_markdown
from src.dashboard.package_builder import build_deployment_package

# Load model
int8_path = PROJECT_ROOT / 'models' / 'quantized' / 'baseline_dscnn_int8.tflite'
with open(int8_path, 'rb') as f:
    int8_bytes = f.read()

analysis = analyze_tflite_bytes(int8_bytes, 'baseline_dscnn_int8.tflite')
compat = check_esp32s3_compatibility(analysis)
compat_md = generate_compatibility_markdown(compat, 'baseline_dscnn_int8')

zip_bytes = build_deployment_package(
    tflite_bytes=int8_bytes,
    tflite_filename='baseline_dscnn_int8.tflite',
    analysis=analysis,
    compatibility_report_md=compat_md,
    json_report=compat.to_dict(),
)

out_zip = PROJECT_ROOT / 'deployment' / 'baseline_dscnn_int8_esp32s3_deploy.zip'
out_zip.write_bytes(zip_bytes)
print(f"Wrote {len(zip_bytes)} bytes to {out_zip}")

# Inspect files
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    manifest = json.loads(zf.read('MANIFEST.json'))
    print("Manifest verification:")
    for a in manifest['artifacts']:
        data = zf.read(a['filename'])
        h = hashlib.sha256(data).hexdigest()
        match = (h == a['sha256']) and (len(data) == a['size_bytes'])
        print(f"  {a['filename']}: {a['size_bytes']} B | SHA-256 match={match}")

    print("\n1. include/model_data.h preview:")
    for l in zf.read('include/model_data.h').decode().splitlines()[:12]:
        print(" ", l)

    print("\n2. include/model_labels.h preview:")
    for l in zf.read('include/model_labels.h').decode().splitlines()[:16]:
        print(" ", l)

    print("\n3. main.cpp preview:")
    for l in zf.read('main.cpp').decode().splitlines()[60:85]:
        print(" ", l)

    print("\n4. compatibility_report.md preview:")
    for l in zf.read('compatibility_report.md').decode().splitlines()[:15]:
        print(" ", l)

    print("\n5. deployment_report.json preview:")
    dep_rep = json.loads(zf.read('deployment_report.json'))
    print("  Deployment status:", dep_rep.get('deployment_status'))
    print("  Resource estimates:", dep_rep.get('resource_estimates'))

    print("\n6. MANIFEST.json preview:")
    print("  Model:", manifest.get('model_name'))
    print("  Total files:", len(manifest.get('artifacts', [])))

    print("\n7. README.md preview:")
    for l in zf.read('README.md').decode().splitlines()[:20]:
        print(" ", l)
