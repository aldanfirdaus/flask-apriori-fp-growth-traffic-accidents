import glob
import os
import nbformat
import numpy as np
from flask import Flask, request, redirect, session, url_for, render_template, flash
import pandas as pd
from werkzeug.utils import secure_filename
from nbconvert.preprocessors import ExecutePreprocessor

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle

app = Flask(__name__)
app.secret_key = 'supersecretkey'
# Mengaktifkan mode debug
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['NOTEBOOK_FOLDER'] = 'notebooks'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Variabel global untuk menyimpan hasil
global_apriori_rules = []
global_fpgrowth_rules = []

def run_notebook(file_path, support, confidence):
    notebook_path = 'notebooks/process_data.ipynb'
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Modify notebook to include input parameters
    for cell in nb.cells:
        if cell.cell_type == 'code' and '## parameters' in cell.source:
            cell.source += f"\nfile_path = '{file_path}'\nsupport = {support}\nconfidence = {confidence}"

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    # Assume processed results are saved in 'processed_results.pkl'
    with open('notebooks/processed_results.pkl', 'rb') as f:
        results_apriori = pickle.load(f)
        results_fpgrowth = pickle.load(f)
        df_gabungan = pickle.load(f)
    return results_apriori, results_fpgrowth, df_gabungan

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def home():
    results_path = os.path.join(app.config['NOTEBOOK_FOLDER'], 'processed_results.pkl')
    if os.path.exists(results_path):
        with open('notebooks/processed_results.pkl', 'rb') as f:
            results_apriori = pickle.load(f)
            results_fpgrowth = pickle.load(f)
            df_gabungan = pickle.load(f)
        # Pastikan kolom 'bulan' ada
        if 'bulan' not in df_gabungan.columns:
            return "Kolom 'bulan' tidak ditemukan dalam DataFrame"
        
        # Menghitung frekuensi munculnya setiap bulan
        bulan_count = df_gabungan['bulan'].value_counts().reset_index()
        bulan_count.columns = ['bulan', 'jumlah']
        # Buat daftar bulan dalam urutan yang diinginkan
        bulan_order = ['JANUARI','FEBRUARI','MARET','APRIL','MEI','JUNI','JULI','AGUSTUS','SEPTEMBER','OKTOBER','NOVEMBER','DESEMBER']
        # Urutkan DataFrame berdasarkan daftar bulan
        bulan_count['bulan'] = pd.Categorical(bulan_count['bulan'], categories=bulan_order, ordered=True)
        bulan_count = bulan_count.sort_values('bulan')
        # Konversi DataFrame ke dictionary agar dapat diserialisasi
        bulan_count_dict = bulan_count.to_dict(orient='list')

        # Hari
        if 'hari' not in df_gabungan.columns:
            return "Kolom 'hari' tidak ditemukan dalam DataFrame"
        # Menghitung frekuensi munculnya setiap hari
        hari_count = df_gabungan['hari'].value_counts().reset_index()
        hari_count.columns = ['hari', 'jumlah']
        # Mengurutkan hari
        hari_order = ['SENIN','SELASA','RABU','KAMIS','JUMAT','SABTU','MINGGU']
        # Urutkan DataFrame berdasarkan daftar hari
        hari_count['hari'] = pd.Categorical(hari_count['hari'], categories=hari_order, ordered=True)
        hari_count = hari_count.sort_values('hari')
        # Konversi DataFrame ke dictionary agar dapat diserialisasi
        hari_count_dict = hari_count.to_dict(orient='list')
        # Waktu 
        if 'waktu' not in df_gabungan.columns:
            return "Kolom 'waktu' tidak ditemukan dalam DataFrame"
        
        # Menghitung frekuensi munculnya setiap waktu
        waktu_count = df_gabungan['waktu'].value_counts().reset_index()
        waktu_count.columns = ['waktu', 'jumlah']
        # Konversi DataFrame ke dictionary agar dapat diserialisasi
        waktu_count_dict = waktu_count.to_dict(orient='list')

        # Geometri
        if 'bentuk geometri' not in df_gabungan.columns:
            return "Kolom 'bentuk geometri' tidak ditemukan dalam DataFrame"
        # Menghitung frekuensi munculnya setiap hari
        geometri_count = df_gabungan['bentuk geometri'].value_counts().reset_index()
        geometri_count.columns = ['geometri', 'jumlah']
        geometri_count_dict = geometri_count.to_dict(orient='list')

        # Tingkat kecelakaan 
        if 'tingkat kecelakaan' not in df_gabungan.columns:
            return "Kolom 'tingkat kecelakaan' tidak ditemukan dalam DataFrame"
        # Menghitung frekuensi munculnya setiap hari
        tKecelakaan_count = df_gabungan['tingkat kecelakaan'].value_counts().reset_index()
        tKecelakaan_count.columns = ['tKecelakaan', 'jumlah']
        # Mengurutkan tKecelakaan
        tKecelakaan_order = ['ringan','sedang','berat']
        # Urutkan DataFrame berdasarkan daftar tKecelakaan
        tKecelakaan_count['tKecelakaan'] = pd.Categorical(tKecelakaan_count['tKecelakaan'], categories=tKecelakaan_order, ordered=True)
        tKecelakaan_count = tKecelakaan_count.sort_values('tKecelakaan')
        # Konversi DataFrame ke dictionary agar dapat diserialisasi
        tKecelakaan_count_dict = tKecelakaan_count.to_dict(orient='list')

        # Tingkat kerugian 
        if 'tingkat kerugian' not in df_gabungan.columns:
            return "Kolom 'tingkat kerugian' tidak ditemukan dalam DataFrame"
        # Menghitung frekuensi munculnya setiap hari
        tKerugian_count = df_gabungan['tingkat kerugian'].value_counts().reset_index()
        tKerugian_count.columns = ['tKerugian', 'jumlah']
        # Mengurutkan tKerugian
        tKerugian_order = ['ringan','sedang','berat']
        # Urutkan DataFrame berdasarkan daftar tKerugian
        tKerugian_count['tKerugian'] = pd.Categorical(tKerugian_count['tKerugian'], categories=tKerugian_order, ordered=True)
        tKerugian_count = tKerugian_count.sort_values('tKerugian')
        # Konversi DataFrame ke dictionary agar dapat diserialisasi
        tKerugian_count_dict = tKerugian_count.to_dict(orient='list')

        # Pihak Terlibat
        if 'pihak terlibat' not in df_gabungan.columns:
            return "Kolom 'pihak terlibat' tidak ditemukan dalam DataFrame"
        # Menghitung frekuensi munculnya setiap hari
        terlibat_count = df_gabungan['pihak terlibat'].value_counts().reset_index()
        terlibat_count.columns = ['terlibat', 'jumlah']
        terlibat_count_dict = terlibat_count.to_dict(orient='list')
         # Convert DataFrame to list of lists
        data = df_gabungan.values.tolist()
        return render_template('index.html', data=data,data_bulan=bulan_count_dict,data_hari=hari_count_dict,data_waktu=waktu_count_dict,
                           data_geometri = geometri_count_dict,data_tKecelakaan=tKecelakaan_count_dict,data_tKerugian = tKerugian_count_dict,
                           data_terlibat = terlibat_count_dict)
    else :
        flash('File telah terhapus! Silahkan upload ulang', 'error')
        return redirect(url_for('proses_data'))
    
@app.route('/proses_data')
def proses_data():
    return render_template('proses_data.html')
@app.route('/informasi_data')
def informasi_data():
    return render_template('informasi_data.html')

after_rules_apriori = []
after_rules_fpgrowth = []
@app.route('/submit', methods=['POST'])
def submit():
    global after_rules_apriori,after_rules_fpgrowth
    support = float(request.form.get('support'))
    confidence = float(request.form.get('confident'))
    file = request.files['file']
    file_path = None
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('proses_data'))
    # Check for existing processed results
    results_path = os.path.join(app.config['NOTEBOOK_FOLDER'], 'processed_results.pkl')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results_apriori = pickle.load(f)
            results_fpgrowth = pickle.load(f)
            df_gabungan = pickle.load(f)
    else:
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('proses_data'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Run the notebook and get results if processed results do not exist
            results_apriori, results_fpgrowth, df_gabungan = run_notebook(file_path, support, confidence)
        else:
            flash('Allowed file types are xlsx')
            return redirect(url_for('proses_data'))
        
    #Proses filtering apriori % keterangan
    after_rules_apriori = results_apriori[(results_apriori['confidence'] >= confidence) & (results_apriori['support'] >= support) ]
    #Proses filtering fpgrowth
    after_rules_fpgrowth = results_fpgrowth[(results_fpgrowth['confidence'] >= confidence) & (results_fpgrowth['support'] >= support) ]
    
    ### Hitung jumlah data kecelakaan, aturan apriori dan fp-growth
    accident_count = len(df_gabungan)
    apriori_rules_total = len(after_rules_apriori)
    fpgrowth_rules_total = len(after_rules_fpgrowth)
    
    apriori_toprules_bysupport = results_apriori.nlargest(1, 'support')
    fpgrowth_toprules_bysupport = results_fpgrowth.nlargest(1, 'support')


    ##Proses menghasilkan keterangan rules
    ket_rules = pd.DataFrame()
    mapping_all = {
        'A1' : 'Minggu','A2' : 'Senin','A3' : 'Selasa','A4' : 'Rabu','A5' : 'Kamis','A6' : 'Jumat','A7' : 'Sabtu',
        'B1' : 'Januari','B2' : 'Februari','B3' : 'Maret','B4' : 'April','B5' :  'Mei','B6' : 'Juni','B7' : 'Juli','B8' : 'Agustus','B9' : 'September','B10' : 'Oktober','B11' : 'November','B12' : 'Desember',
        'C1' : 'Dini Hari','C2' : 'Pagi','C3' : 'Siang','C4' : 'Sore','C5' : 'Petang','C6' : 'Malam',
        'D1' : 'jalan lurus','D2' : 'bundaran','D3' : 'simpang 3','D4' : 'simpang 4','D5' : 'jembatan','D6' : 'rel kereta api',
        'E1' : 'R2 X R2 X R2','E2' : 'R2 X R4 X R4','E3' : 'R2 X R2 X BUS','E4' : 'R4 X R4 X BUS','E5' : 'R4 X R2 X R2',
        'E6' : 'R2 X R4 X TRUK',
        'E7' : 'R2 X R2',
        'E8' : 'R2 X R4',
        'E9' : 'R4 X R4',
        'E10' : 'R2 X FORKLIFT',
        'E11' : 'R2 X BECAK MOTOR',
        'E12' : 'R2 X PEJALAN KAKI',
        'E13' : 'R4 X PEJALAN KAKI',
        'E14' : 'R2 X KERETA API',
        'E15' : 'R2 X BUS',
        'E16' : 'R2 X TRUK',
        'E17' : 'R4 X TRUK',
        'E18' : 'R3 X TRUK',
        'E19' : 'TRUK X TRUK',
        'E20' : 'SEPEDA ANGIN X R4',
        'E21' : 'SEPEDA ANGIN X R2',
        'E22' : 'R2 X R3',
        'E23' : 'R3 X R3',
        'E24' : 'R2',
        'E25' : 'R3',
        'E26' : 'R4',
        'E27' : 'R2 X BECAK',
        'F1' : 'Tingkat kecelakaan ringan',
        'F2' : 'Tingkat kecelakaan sedang',
        'F3' : 'Tingkat kecelakaan berat',
        'G1' : 'Tingkat kerugian kecelakaan ringan',
        'G2' : 'Tingkat kerugian kecelakaan sedang',
        'G3' : 'Tingkat kerugian kecelakaan berat'
    }
    top_apriori = pd.DataFrame()
    top_fpgrowth = pd.DataFrame()
    
    def map_values(value_set):
        return frozenset(mapping_all[item] for item in value_set)

    top_apriori['antecedents'] = apriori_toprules_bysupport['antecedents'].apply(map_values)
    top_apriori['consequents'] = apriori_toprules_bysupport['consequents'].apply(map_values)
    top_apriori['confident'] = (apriori_toprules_bysupport['confidence'] * 100).apply(lambda x: f"{x:.2f}%")

    top_fpgrowth['antecedents'] = fpgrowth_toprules_bysupport['antecedents'].apply(map_values)
    top_fpgrowth['consequents'] = fpgrowth_toprules_bysupport['consequents'].apply(map_values)
    top_fpgrowth['confident'] = (fpgrowth_toprules_bysupport['confidence'] * 100).apply(lambda x: f"{x:.2f}%")

    top_apriori = top_apriori.to_dict(orient='records')
    top_fpgrowth = top_fpgrowth.to_dict(orient='records')

    def convert_frozenset_top_apriori(top_apriori):
        top_apriori['antecedents'] = ','.join(list(top_apriori['antecedents']))
        top_apriori['consequents'] = ','.join(list(top_apriori['consequents']))
        return top_apriori
    def convert_frozenset_top_fpgrowth(top_fpgrowth):
        top_fpgrowth['antecedents'] = ','.join(list(top_fpgrowth['antecedents']))
        top_fpgrowth['consequents'] = ','.join(list(top_fpgrowth['consequents']))
        return top_fpgrowth
    
    top_apriori = [convert_frozenset_top_apriori(top_apriori) for top_apriori in top_apriori]
    top_fpgrowth = [convert_frozenset_top_fpgrowth(top_fpgrowth) for top_fpgrowth in top_fpgrowth]

    # Function to map the values using the mapping dictionary
    def map_values(value_set):
        return frozenset(mapping_all[item] for item in value_set)
    # Apply the mapping function to the 'antecedents' and 'consequents' columns
    ket_rules['antecedents'] = after_rules_apriori['antecedents'].apply(map_values)
    ket_rules['consequents'] = after_rules_apriori['consequents'].apply(map_values)
    ket_rules.insert(0, 'no', range(1, len(ket_rules) + 1))
    
    # Add the 'confidence' column from 'rules_apriori' directly
    ket_rules['confidence'] = (after_rules_apriori['confidence'] * 100).apply(lambda x: f"{x:.2f}%")
    
    #Proses merubah menjadi list apriori & fpgrowth
    data_list_apriori = after_rules_apriori.to_dict(orient='records')
    data_list_fpgrowth = after_rules_fpgrowth.to_dict(orient='records')
    ket_rules = ket_rules.to_dict(orient='records')
    # data_list_ketRules = ket_rules.to_dict(orient='list')
    def convert_frozenset_ket(ket):
        ket['antecedents'] = ','.join(list(ket['antecedents']))
        ket['consequents'] = ','.join(list(ket['consequents']))
        return ket
    def convert_frozenset(row):
        row['antecedents'] = ','.join(list(row['antecedents']))
        row['consequents'] = ','.join(list(row['consequents']))
        row['support'] = round(row['support'], 3)
        row['confidence'] = round(row['confidence'], 3)
        row['lift'] = round(row['lift'], 3)
        return row

    data_list_apriori = [convert_frozenset(row) for row in data_list_apriori]
    data_list_fpgrowth = [convert_frozenset(row) for row in data_list_fpgrowth] 
    ket_rules = [convert_frozenset_ket(ket) for ket in ket_rules]

    ### Tabel persamaan rules kedua algoritma
    # Mencari aturan yang sama berdasarkan 'antecedents' dan 'consequents'
    common_rules_apriori = pd.DataFrame()
    common_rules_fpgrowth = pd.DataFrame()
    def map_values(value_set):
        return frozenset(mapping_all[item] for item in value_set)
    # Apply the mapping function to the 'antecedents' and 'consequents' columns
    common_rules_apriori['antecedents'] = after_rules_apriori['antecedents'].apply(map_values)
    common_rules_apriori['consequents'] = after_rules_apriori['consequents'].apply(map_values)
    common_rules_apriori['support_apriori'] = (after_rules_apriori['support'] * 100).apply(lambda x: f"{x:.2f}%")
    common_rules_apriori['confidence_apriori'] = (after_rules_apriori['confidence'] * 100).apply(lambda x: f"{x:.2f}%")

    common_rules_fpgrowth['antecedents'] = after_rules_fpgrowth['antecedents'].apply(map_values)
    common_rules_fpgrowth['consequents'] = after_rules_fpgrowth['consequents'].apply(map_values)
    common_rules_fpgrowth['support_fpgrowth'] = (after_rules_fpgrowth['support'] * 100).apply(lambda x: f"{x:.2f}%")
    common_rules_fpgrowth['confidence_fpgrowth'] = (after_rules_fpgrowth['confidence'] * 100).apply(lambda x: f"{x:.2f}%")

    common_rules = pd.merge(
       common_rules_apriori, 
        common_rules_fpgrowth, 
        on=['antecedents', 'consequents'], 
        how='inner'  # Hanya aturan yang sama yang akan diambil
    )
    common_rules_list = common_rules.to_dict(orient='records')
    def convert_frozenset_common_rules_list(common_rules_list):
        common_rules_list['antecedents'] = ','.join(list(common_rules_list['antecedents']))
        common_rules_list['consequents'] = ','.join(list(common_rules_list['consequents']))
        return common_rules_list
    
    common_rules_list = [convert_frozenset_common_rules_list(common_rules_list) for common_rules_list in common_rules_list]


    #### faktor kecelakaan
    # Dictionary untuk menyimpan tabel rules dari faktor A-G
    factors = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    rules_by_factor_apriori = {}
    count_factor_apriori = {}

    rules_by_factor_fpgrowth = {}
    count_factor_fpgrowth = {}
    # Loop untuk memproses setiap faktor kecelakaan Apriori
    for factor in factors:
        filtered_rules_apriori = after_rules_apriori[
            after_rules_apriori['antecedents'].apply(lambda x: any(item.startswith(factor) for item in x)) |
            after_rules_apriori['consequents'].apply(lambda x: any(item.startswith(factor) for item in x))
        ]
        filtered_rules_fpgrowth = after_rules_fpgrowth[
            after_rules_fpgrowth['antecedents'].apply(lambda x: any(item.startswith(factor) for item in x)) |
            after_rules_fpgrowth['consequents'].apply(lambda x: any(item.startswith(factor) for item in x))
        ]

        # Format ulang data untuk tabel
        filtered_rules_apriori['antecedents'] = filtered_rules_apriori['antecedents'].apply(map_values)
        filtered_rules_apriori['consequents'] = filtered_rules_apriori['consequents'].apply(map_values)
        filtered_rules_apriori['support_apriori'] = (filtered_rules_apriori['support'] * 100).apply(lambda x: f"{x:.2f}%")
        filtered_rules_apriori['confidence_apriori'] = (filtered_rules_apriori['confidence'] * 100).apply(lambda x: f"{x:.2f}%")

        # Format ulang data untuk tabel
        filtered_rules_fpgrowth['antecedents'] = filtered_rules_fpgrowth['antecedents'].apply(map_values)
        filtered_rules_fpgrowth['consequents'] = filtered_rules_fpgrowth['consequents'].apply(map_values)
        filtered_rules_fpgrowth['support_fpgrowth'] = (filtered_rules_fpgrowth['support'] * 100).apply(lambda x: f"{x:.2f}%")
        filtered_rules_fpgrowth['confidence_fpgrowth'] = (filtered_rules_fpgrowth['confidence'] * 100).apply(lambda x: f"{x:.2f}%")
        
         # Konversi frozenset ke string dalam list of dictionaries
        def convert_frozenset_to_string(rule):
            rule['antecedents'] = ', '.join(list(rule['antecedents']))
            rule['consequents'] = ', '.join(list(rule['consequents']))
            return rule

        # Konversi filtered_rules menjadi list of dictionaries
        rules_list_apriori = filtered_rules_apriori.to_dict(orient='records')
        rules_list_apriori = [convert_frozenset_to_string(rule) for rule in rules_list_apriori]
        rules_by_factor_apriori[factor] = rules_list_apriori
        count_factor_apriori[factor] = len(rules_list_apriori)

        rules_list_fpgrowth = filtered_rules_fpgrowth.to_dict(orient='records')
        rules_list_fpgrowth = [convert_frozenset_to_string(rule) for rule in rules_list_fpgrowth]
        rules_by_factor_fpgrowth[factor] = rules_list_fpgrowth
        count_factor_fpgrowth[factor] = len(rules_list_fpgrowth)
    # Loop untuk memproses setiap faktor kecelakaan

    ## Percobaan

    def generate_description(row):
        # Fungsi untuk memformat item menjadi string dengan "dan" di akhir
        def format_items(items):
            items = [item.strip() for item in items]  # Membersihkan spasi
            if len(items) == 1:
                return items[0]
            return ', '.join(items[:-1]) + ' dan ' + items[-1]

        # Fungsi untuk membangun deskripsi berdasarkan urutan prioritas
        def build_description(items):
            description = []
            remaining_items = []

            for item in items:
                item = item.strip()
                if item in ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]:
                    description.append(f"kecelakaan terjadi pada hari {item}")
                elif item in ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]:
                    description.append(f"kecelakaan terjadi pada bulan {item}")
                elif item in ["Dini Hari", "Pagi", "Siang", "Sore", "Petang", "Malam"]:
                    description.append(f"kecelakaan terjadi {item.lower()}")
                elif item in ["jalan lurus", "bundaran", "simpang", "jembatan", "rel"]:
                    description.append(f"kecelakaan terjadi di {item.lower()}")
                elif item in ["R2 X R2 X R2", "R2 X R4 X R4","R2 X R2 X BUS", "R4 X R4 X BUS","R4 X R2 X R2","R2 X R4 X TRUK","R2 X R2", "R2 X R4","R4 X R4","R2 X FORKLIFT", "R2 X BECAK MOTOR", "R2 X PEJALAN KAKI","R4 X PEJALAN KAKI", "R2 X KERETA API", "R2 X BUS", "R2 X TRUK", "R4 X TRUK", "R3 X TRUK", "TRUK X TRUK", "SEPEDA ANGIN X R4", "SEPEDA ANGIN X R2","R2 X R3", "R3 X R3", "R2", "R3", "R4", "R2 X BECAK"]:
                    description.append(f"kecelakaan melibatkan {item}")
                elif "Tingkat" in item:
                    description.append(f"{item.lower()}")
                else:
                    remaining_items.append(item)

            # Tambahkan item lainnya jika ada
            if remaining_items:
                description.append(format_items(remaining_items))

            return ' dan '.join(description)
    
        # Bagian antecedents
        antecedents_items = row['antecedents'].split(',')
        antecedents_desc = build_description(antecedents_items)

        # Bagian consequents
        consequents_items = row['consequents'].split(',')
        consequents_desc = build_description(consequents_items)

        # Gabungkan antecedents dan consequents
        return f"Jika {antecedents_desc} maka {consequents_desc}."

    # Terapkan fungsi ke DataFrame
    for rule in common_rules_list:
        rule['description'] = generate_description(rule)
    # Menambahkan deskripsi ke setiap faktor kecelakaan
    for factor, rules_list_apriori in rules_by_factor_apriori.items():
        for rule in rules_list_apriori:
            rule['description'] = generate_description(rule)
    # Menambahkan deskripsi ke setiap faktor kecelakaan
    for factor, rules_list_fpgrowth in rules_by_factor_fpgrowth.items():
        for rule in rules_list_fpgrowth:
            rule['description'] = generate_description(rule)
    ## Percobaan
    
    # Network Graph Apriori...............................................................
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # Inisialisasi directed graph
    G = nx.DiGraph()

    # Loop melalui rules_apriori untuk menambahkan edges
    for _, row in after_rules_apriori.iterrows():
        antecedents = ', '.join([mapping_all.get(item, item) for item in list(row['antecedents'])])  # Mengonversi frozenset ke string
        consequents = ', '.join([mapping_all.get(item, item) for item in list(row['consequents'])])
        confidence = row['confidence']
        lift = row['lift']
        
        # Tambahkan edge ke grafik
        G.add_edge(antecedents, consequents, weight=confidence, lift=lift)

    # Menentukan posisi layout untuk nodes
    pos = nx.spring_layout(G)
    # Visualisasi grafik
    plt.figure(figsize=(15, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    # Tambahkan label pada edges untuk menampilkan confidence dan lift
    # edge_labels = { (', '.join(list(row['antecedents'])), ', '.join(list(row['consequents']))): f"Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f}" for _, row in rules_apriori.iterrows()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Network Graph Asosiasi Produk")
    plt.savefig('static/assets/img/chartGraphApriori.png')  # Simpan grafik sebagai gambar
    # plt.close()  # Tutup plot untuk menghindari tampilan berulang
    # Network Graph...............................................................

    # Network Graph FpGrowth...............................................................
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # Inisialisasi directed graph
    G = nx.DiGraph()

    # Loop melalui rules_apriori untuk menambahkan edges
    for _, row in after_rules_fpgrowth.iterrows():
        antecedents = ', '.join([mapping_all.get(item, item) for item in list(row['antecedents'])])  # Mengonversi frozenset ke string
        consequents = ', '.join([mapping_all.get(item, item) for item in list(row['consequents'])])
        confidence = row['confidence']
        lift = row['lift']
        
        # Tambahkan edge ke grafik
        G.add_edge(antecedents, consequents, weight=confidence, lift=lift)

    # Menentukan posisi layout untuk nodes
    pos = nx.spring_layout(G)
    # Visualisasi grafik
    plt.figure(figsize=(15, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    # Tambahkan label pada edges untuk menampilkan confidence dan lift
    # edge_labels = { (', '.join(list(row['antecedents'])), ', '.join(list(row['consequents']))): f"Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f}" for _, row in rules_apriori.iterrows()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Network Graph Asosiasi FP-Growth")
    plt.savefig('static/assets/img/chartGraphFpgrowth.png')  # Simpan grafik sebagai gambar
    # plt.close()  # Tutup plot untuk menghindari tampilan berulang
    # Network Graph...............................................................

    # Bar Chart Support...................................................................
    # Fungsi untuk mengganti kode dengan deskripsi
    def replace_with_descriptions(frozenset_items):
        return ', '.join([mapping_all.get(item, item) for item in frozenset_items])
    
    top_rules_apriori = after_rules_apriori.nlargest(5, 'support')
    top_rules_fpgrowth = after_rules_fpgrowth.nlargest(5, 'support')

    # Konversi frozenset ke deskripsi
    top_rules_apriori['rule'] = top_rules_apriori['antecedents'].apply(lambda x: replace_with_descriptions(x)) + ' → ' + \
                                top_rules_apriori['consequents'].apply(lambda x: replace_with_descriptions(x))

    top_rules_fpgrowth['rule'] = top_rules_fpgrowth['antecedents'].apply(lambda x: replace_with_descriptions(x)) + ' → ' + \
                                top_rules_fpgrowth['consequents'].apply(lambda x: replace_with_descriptions(x))


    # Gabungkan kedua dataset untuk memastikan aturan yang sama diurutkan
    all_rules = pd.DataFrame({
        'rule': pd.concat([top_rules_apriori['rule'], top_rules_fpgrowth['rule']]).unique(),
        'Apriori': 0,
        'FP-Growth': 0
    })
   

    # Isi nilai support untuk Apriori dan FP-Growth
    for _, row in top_rules_apriori.iterrows():
        all_rules.loc[all_rules['rule'] == row['rule'], 'Apriori'] = row['support']

    for _, row in top_rules_fpgrowth.iterrows():
        all_rules.loc[all_rules['rule'] == row['rule'], 'FP-Growth'] = row['support']
    # Plot Side-by-Side Bar Chart
    bar_width = 0.4  # Lebar setiap bar
    x = np.arange(len(all_rules))  # Posisi x untuk setiap rule

    plt.figure(figsize=(12, 8))
    plt.bar(x - bar_width / 2, all_rules['Apriori'], bar_width, label='Apriori', color='skyblue')
    plt.bar(x + bar_width / 2, all_rules['FP-Growth'], bar_width, label='FP-Growth', color='lightgreen')

    # Tambahkan label dan judul
    plt.xticks(x, all_rules['rule'], rotation=45, ha='right', fontsize=10)
    plt.xlabel('Rules')
    plt.ylabel('Support')
    plt.title('Comparison of Top 5 Rules by Support (Apriori vs FP-Growth)')
    plt.legend()

    # Simpan grafik
    plt.tight_layout()
    plt.savefig('static/assets/img/chartbar_combined.png')  # Simpan grafik sebagai gambar
    plt.show()

    # Bar Chart Confidence...................................................................
    def replace_with_descriptions(frozenset_items):
        return ', '.join([mapping_all.get(item, item) for item in frozenset_items])
    
    top_rules_apriori = after_rules_apriori.nlargest(5, 'confidence')
    top_rules_fpgrowth = after_rules_fpgrowth.nlargest(5, 'confidence')

    # Konversi frozenset ke deskripsi
    top_rules_apriori['rule'] = top_rules_apriori['antecedents'].apply(lambda x: replace_with_descriptions(x)) + ' → ' + \
                                top_rules_apriori['consequents'].apply(lambda x: replace_with_descriptions(x))

    top_rules_fpgrowth['rule'] = top_rules_fpgrowth['antecedents'].apply(lambda x: replace_with_descriptions(x)) + ' → ' + \
                                top_rules_fpgrowth['consequents'].apply(lambda x: replace_with_descriptions(x))


    # Gabungkan kedua dataset untuk memastikan aturan yang sama diurutkan
    all_rules = pd.DataFrame({
        'rule': pd.concat([top_rules_apriori['rule'], top_rules_fpgrowth['rule']]).unique(),
        'Apriori': 0,
        'FP-Growth': 0
    })
   
    # Isi nilai confidence untuk Apriori dan FP-Growth
    for _, row in top_rules_apriori.iterrows():
        all_rules.loc[all_rules['rule'] == row['rule'], 'Apriori'] = row['confidence']

    for _, row in top_rules_fpgrowth.iterrows():
        all_rules.loc[all_rules['rule'] == row['rule'], 'FP-Growth'] = row['confidence']
    # Plot Side-by-Side Bar Chart
    bar_width = 0.4  # Lebar setiap bar
    x = np.arange(len(all_rules))  # Posisi x untuk setiap rule

    plt.figure(figsize=(12, 8))
    plt.bar(x - bar_width / 2, all_rules['Apriori'], bar_width, label='Apriori', color='skyblue')
    plt.bar(x + bar_width / 2, all_rules['FP-Growth'], bar_width, label='FP-Growth', color='lightgreen')

    # Tambahkan label dan judul
    plt.xticks(x, all_rules['rule'], rotation=45, ha='right', fontsize=10)
    plt.xlabel('Rules')
    plt.ylabel('confidence')
    plt.title('Comparison of Top 5 Rules by Confidence (Apriori vs FP-Growth)')
    plt.legend()

    # Simpan grafik
    plt.tight_layout()
    plt.savefig('static/assets/img/chartbar_confidence.png')  # Simpan grafik sebagai gambar
    plt.show()
    # Bar Chart Confidence...................................................................

    # Pie Chart...................................................................
    # Membuat Pie Chart untuk Confidence
    # plt.figure(figsize=(8, 8))
    # plt.pie(top_rules_confidence['confidence'], labels=top_rules_confidence['antecedents'] + ' → ' + top_rules_confidence['consequents'], 
    #         autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    # plt.title('Top 5 Rules Berdasarkan Confidence')
    # plt.savefig('static/assets/img/chartpie.png')  # Simpan grafik sebagai gambar
    # plt.close()  # Tutup plot untuk menghindari tampilan berulang

    show_result = True  # Penanda bahwa hasil harus ditampilkan


    return render_template('proses_data.html', rules_apriori=data_list_apriori, rules_fpgrowth=data_list_fpgrowth,ket_rules = ket_rules, show_result=show_result,accident_count=accident_count,
                           apriori_rules_total=apriori_rules_total,fpgrowth_rules_total=fpgrowth_rules_total,top_apriori=top_apriori,top_fpgrowth=top_fpgrowth,
                           common_rules = common_rules_list,rules_by_factor_apriori=rules_by_factor_apriori,rules_by_factor_fpgrowth=rules_by_factor_fpgrowth,
                           count_factor_apriori=count_factor_apriori,count_factor_fpgrowth=count_factor_fpgrowth)




@app.route('/delete', methods=['POST'])
def delete_files():
    # Delete all files in the uploads folder
    upload_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for file_path in upload_files:
        os.remove(file_path)

    # Delete all pickle files in the notebooks folder
    pickle_files = glob.glob(os.path.join(app.config['NOTEBOOK_FOLDER'], '*.pkl'))
    for file_path in pickle_files:
        os.remove(file_path)

    flash('File telah berhasil dihapus')
    return redirect(url_for('proses_data'))

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('notebooks'):
        os.makedirs('notebooks')
    app.run(debug=True)