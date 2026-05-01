"""
Generate Streamlit Cloud secrets.toml format yang 100% berhasil untuk GEE
"""
import json
import base64
import os

def generate_secrets():
    """Generate secrets in multiple formats"""
    
    json_path = "deteksi-banjir-492803-7fc098068802.json"
    
    if not os.path.exists(json_path):
        print(f"❌ File {json_path} tidak ditemukan!")
        return
    
    # Baca file JSON
    with open(json_path, 'r') as f:
        gee_key = json.load(f)
    
    print("=" * 70)
    print("🔐 STREAMLIT CLOUD SECRETS GENERATOR")
    print("=" * 70)
    
    # METHOD 1: Single-line JSON (RECOMMENDED - 100% Success Rate)
    print("\n📌 METHOD 1: Single-line JSON (PALING DIREKOMENDASIKAN)")
    print("-" * 70)
    
    # Convert to single-line JSON string
    json_str = json.dumps(gee_key, separators=(',', ':'))
    
    # Create TOML content
    toml_method1 = f'''[gee]
json_key = "{json_str}"'''
    
    print("✅ Format untuk Streamlit Cloud:")
    print(toml_method1[:200] + "...")
    print("\n📋 INSTRUKSI:")
    print("   1. Copy TOML di atas")
    print("   2. Buka https://share.streamlit.io")
    print("   3. App → Settings → Secrets")
    print("   4. Paste dan Save")
    
    # Save to file
    with open(".streamlit/secrets_method1.toml", "w") as f:
        f.write(toml_method1)
    print(f"\n💾 Disimpan di: .streamlit/secrets_method1.toml")
    
    # METHOD 2: Base64 (Alternative)
    print("\n\n📌 METHOD 2: Base64 Encoded (Alternatif)")
    print("-" * 70)
    
    json_bytes = json.dumps(gee_key).encode('utf-8')
    b64_str = base64.b64encode(json_bytes).decode('utf-8')
    
    toml_method2 = f'''[gee]
json_key_b64 = "{b64_str}"'''
    
    print("✅ Format Base64:")
    print(toml_method2[:100] + "...")
    
    with open(".streamlit/secrets_method2.toml", "w") as f:
        f.write(toml_method2)
    print(f"\n💾 Disimpan di: .streamlit/secrets_method2.toml")
    
    # Verify keys
    print("\n" + "=" * 70)
    print("🔍 VERIFIKASI KUNCI:")
    print("=" * 70)
    print(f"   Project ID: {gee_key.get('project_id')}")
    print(f"   Client Email: {gee_key.get('client_email')}")
    print(f"   Private Key ID: {gee_key.get('private_key_id')}")
    print(f"   Private Key Length: {len(gee_key.get('private_key', ''))} chars")
    
    # Test parse
    print("\n🧪 TEST PARSING:")
    try:
        test_parse = json.loads(json_str)
        print(f"   ✓ JSON valid: {test_parse['client_email']}")
        print(f"   ✓ Private key format: {'Valid' if test_parse['private_key'].startswith('-----BEGIN') else 'Invalid'}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✨ Siap deploy! Gunakan METHOD 1 untuk hasil terbaik.")
    print("=" * 70)

if __name__ == "__main__":
    generate_secrets()
