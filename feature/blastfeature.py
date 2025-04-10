import subprocess
import tempfile
import os
def get_max_identity(sequence, db):
    """
    Given an amino acid sequence, run BLASTp alignment and return the maximum identity.

    Parameters:
        sequence (str): The amino acid sequence to query.

    Returns:
        max_identity (float): The maximum identity from the BLASTp alignment.
    """
    # Create a temporary file to store the query sequence in FASTA format
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_fasta:
        query_fasta = temp_fasta.name
        with open(query_fasta, "w") as f:
            f.write(f">query_seq\n{sequence}\n")

    # Configure the BLASTp command
    output_file = "tmp/blast_results.txt"
    blastp_cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", f"database/{db}_db",  # Use custom database
        "-evalue", "1e-5",
        "-outfmt", "6",  # Output in tabular format (outfmt 6)
        "-out", output_file
    ]

    # Run the BLASTp command
    subprocess.run(blastp_cmd)

    # Initialize the maximum identity
    max_identity = 0

    # Read the BLASTp output file and extract identity
    with open(output_file) as f:
        for line in f:
            columns = line.strip().split("\t")
            identity = float(columns[2])  # The third column is %identity
            if identity > max_identity:
                max_identity = identity

    # Clean up the temporary file
    os.remove(query_fasta)
    os.remove(output_file)
    return max_identity
