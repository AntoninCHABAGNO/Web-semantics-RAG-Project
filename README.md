# Web-datamining-semantics-Project-GOT

conda create -n semantic-web-data-mining-project python=3.12

conda activate semantic-web-data-mining-project

pip install -r requirements.txt
pip install openai python-dotenv

Ordre d'exécution :

python src/A_01_crawl_clean.py

python src/B_02_nlp_extract.py --report

python src/C_03_clean_before_rdf.py
python src/C_04_csv_to_rdf.py

python src/D_05_predicate_alignment.py
python src/D_06_entity_linking.py
python src/D_07_filter_entity_linking.py
python src/D_08_expand_kb.py

python src/E_09_prepare_triples.py
python src/E_10_clean_for_embedding.py
python src/E_11_split_dataset.py
python src/E_12_train_eval_visualize_kge.py

python src/F_13_build_chunks.py
python src/F_14_build_vector_index.py
python src/F_15_text_retriever.py --question "Who is Beth Harmon?" --top_k 5
python src/F_16_kg_retriever.py --question "Who is Beth Harmon?"

python src/F_17_hybrid_retriever_reranked.py --question "Who is Beth Harmon?" --output_json data/rag/evidence_beth.json
python src/F_18_generate_answer.py --question "Who is Beth Harmon?" --evidence_json data/rag/evidence_beth.json
python src/F_19_rag_cli.py --question "Who is Beth Harmon?" --mode debug