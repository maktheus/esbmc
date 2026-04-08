#!/usr/bin/env python3
import os

def bin_to_c_array(bin_filepath, output_filepath, array_name="model_data"):
    if not os.path.exists(bin_filepath):
        print(f"Erro: O arquivo {bin_filepath} não existe.")
        return

    with open(bin_filepath, "rb") as f:
        data = f.read()

    with open(output_filepath, "w") as f:
        f.write("/* Arquivo gerado automaticamente - NÃO EDITE */\n")
        f.write(f"/* Contém os bytes do modelo ({len(data)} bytes) mapeados estaticamente para o ESBMC */\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        # Format bytes as hex
        hex_bytes = [f"0x{b:02x}" for b in data]
        
        # Group by 12 bytes per line
        lines = []
        for i in range(0, len(hex_bytes), 12):
            chunk = hex_bytes[i:i+12]
            lines.append("    " + ", ".join(chunk) + ",")
            
        f.write("\n".join(lines))
        f.write("\n};\n")

    print(f"Sucesso: Array C gerado em {output_filepath} com array de bytes '{array_name}'.")

if __name__ == "__main__":
    # Called from project root
    bin_to_c_array("src/dummy_model.bin", "src/model_data.h")
