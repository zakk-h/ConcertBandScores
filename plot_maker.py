import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_scores(filenames):
    results = []

    for file in filenames:
        year = file[:4]
        df = pd.read_csv(file)
        
        roman_to_int = {
            'I': 1,
            'II': 2,
            'III': 3,
            'IV': 4,
            'V': 5,
            'VI': 6,
            'I/II': 1.5,  
            'II/III': 2.5,
            'III/IV': 3.5,
            'IV/V': 4.5,
            'V/VI': 5.5,
            'VI/MW': 6.5, 
            'MW': 7, 
            'co': None, 
            'C/O': None 
        }
        
        df['Judge 1'] = df['Judge 1'].map(roman_to_int)
        df['Judge 2'] = df['Judge 2'].map(roman_to_int)
        df['Judge 3'] = df['Judge 3'].map(roman_to_int)
        df['Grade'] = df['Grade'].map(roman_to_int) 
        
        df['Total Score'] = df[['Judge 1', 'Judge 2', 'Judge 3']].sum(axis=1)
        
        df.dropna(subset=['Total Score'], inplace=True)
        
        districts = df['District'].unique()
        
        for district in districts:
            df_district = df[df['District'] == district]
            counts = df_district['Total Score'].value_counts().sort_index()
            proportions = (counts / counts.sum()).sort_index()
            avg_score = df_district['Total Score'].mean()
            avg_grade = df_district['Grade'].mean()
            grade_counts = df_district['Grade'].value_counts(normalize=True).sort_index()
            results.append({
                'Year': year, 'District': district, 'Counts': proportions.to_dict(), 
                'Average Score': avg_score, 'Average Grade': avg_grade, 'Grade Proportions': grade_counts.to_dict()
            })
        
        overall_counts = df['Total Score'].value_counts().sort_index()
        overall_proportions = (overall_counts / overall_counts.sum()).sort_index()
        overall_avg = df['Total Score'].mean()
        overall_avg_grade = df['Grade'].mean()
        overall_grade_counts = df['Grade'].value_counts(normalize=True).sort_index()
        results.append({
            'Year': year, 'District': 'Overall', 'Counts': overall_proportions.to_dict(), 
            'Average Score': overall_avg, 'Average Grade': overall_avg_grade, 'Grade Proportions': overall_grade_counts.to_dict()
        })
    
    return results

def plot_histograms(results):
    if not os.path.exists('histograms'):
        os.makedirs('histograms')
    
    for result in results:
        year = result['Year']
        district = result['District']
        counts = result['Counts']
        
        plt.figure()
        plt.bar(counts.keys(), counts.values(), color='skyblue')
        plt.xlabel('Total Score')
        plt.ylabel('Proportion')
        plt.title(f'Distribution of Total Scores in {district} ({year})')
        plt.xticks(range(3, 16))
        plt.grid(axis='y')
        plt.savefig(f'histograms/{district}_{year}.png')
        plt.close()

def plot_mean_scores_per_year(results):
    district_year_avg_scores = {}

    for result in results:
        year = result['Year']
        district = result['District']
        avg_score = result['Average Score']
        
        if district not in district_year_avg_scores:
            district_year_avg_scores[district] = {}
        district_year_avg_scores[district][year] = avg_score

    if not os.path.exists('mean_scores'):
        os.makedirs('mean_scores')
    
    for district, years in district_year_avg_scores.items():
        plt.figure()
        plt.bar(years.keys(), years.values(), color='skyblue')
        plt.xlabel('Year')
        plt.ylabel('Average Score')
        plt.title(f'Average Scores in {district} Over Years')
        plt.grid(axis='y')
        plt.savefig(f'mean_scores/{district}_mean_scores.png')
        plt.close()

def plot_proportions_per_score(results):
    district_year_proportions = {}

    for result in results:
        year = result['Year']
        district = result['District']
        counts = result['Counts']
        
        if district not in district_year_proportions:
            district_year_proportions[district] = {}
        
        for score, proportion in counts.items():
            if score not in district_year_proportions[district]:
                district_year_proportions[district][score] = {}
            district_year_proportions[district][score][year] = proportion
    
    if not os.path.exists('proportions'):
        os.makedirs('proportions')
    
    for district, scores in district_year_proportions.items():
        for score, years in scores.items():
            plt.figure()
            plt.bar(years.keys(), years.values(), color='skyblue')
            plt.xlabel('Year')
            plt.ylabel('Proportion')
            plt.title(f'Proportion of Score {score} in {district} Over Years')
            plt.grid(axis='y')
            plt.savefig(f'proportions/{district}_score_{score}_proportions.png')
            plt.close()

def plot_mean_grades_per_year(results):
    district_year_avg_grades = {}

    for result in results:
        year = result['Year']
        district = result['District']
        avg_grade = result['Average Grade']
        
        if district not in district_year_avg_grades:
            district_year_avg_grades[district] = {}
        district_year_avg_grades[district][year] = avg_grade

    if not os.path.exists('mean_grades'):
        os.makedirs('mean_grades')
    
    for district, years in district_year_avg_grades.items():
        plt.figure()
        plt.bar(years.keys(), years.values(), color='skyblue')
        plt.xlabel('Year')
        plt.ylabel('Average Grade')
        plt.title(f'Average Grades in {district} Over Years')
        plt.grid(axis='y')
        plt.savefig(f'mean_grades/{district}_mean_grades.png')
        plt.close()

def plot_proportions_per_grade(results):
    district_year_grade_proportions = {}

    for result in results:
        year = result['Year']
        district = result['District']
        grade_proportions = result['Grade Proportions']
        
        if district not in district_year_grade_proportions:
            district_year_grade_proportions[district] = {}
        
        for grade, proportion in grade_proportions.items():
            if grade not in district_year_grade_proportions[district]:
                district_year_grade_proportions[district][grade] = {}
            district_year_grade_proportions[district][grade][year] = proportion
    
    if not os.path.exists('grade_proportions'):
        os.makedirs('grade_proportions')
    
    for district, grades in district_year_grade_proportions.items():
        for grade, years in grades.items():
            plt.figure()
            plt.bar(years.keys(), years.values(), color='skyblue')
            plt.xlabel('Year')
            plt.ylabel('Proportion')
            plt.title(f'Proportion of Grade {grade} in {district} Over Years')
            plt.grid(axis='y')
            plt.savefig(f'grade_proportions/{district}_grade_{grade}_proportions.png')
            plt.close()

def print_results(results):
    year_avg_scores = {}
    district_year_avg_scores = {}

    year_avg_grades = {}
    district_year_avg_grades = {}

    for result in results:
        year = result['Year']
        district = result['District']
        avg_score = result['Average Score']
        avg_grade = result['Average Grade']
        
        if year not in year_avg_scores:
            year_avg_scores[year] = []
        year_avg_scores[year].append(avg_score)
        
        if district not in district_year_avg_scores:
            district_year_avg_scores[district] = {}
        if year not in district_year_avg_scores[district]:
            district_year_avg_scores[district][year] = avg_score

        if year not in year_avg_grades:
            year_avg_grades[year] = []
        year_avg_grades[year].append(avg_grade)

        if district not in district_year_avg_grades:
            district_year_avg_grades[district] = {}
        if year not in district_year_avg_grades[district]:
            district_year_avg_grades[district][year] = avg_grade
        
        print(f"Year: {year}, District: {district}")
        print(f"Score Counts: {result['Counts']}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Grade: {avg_grade:.2f}")
        print("\n")

    print("Average Scores per Year:")
    for year, scores in year_avg_scores.items():
        print(f"{year}: {sum(scores) / len(scores):.2f}")
    print("\n")

    print("Average Scores per District per Year:")
    for district, years in district_year_avg_scores.items():
        for year, avg_score in years.items():
            print(f"District: {district}, Year: {year}, Average Score: {avg_score:.2f}")
    print("\n")

    print("Average Grades per Year:")
    for year, grades in year_avg_grades.items():
        print(f"{year}: {sum(grades) / len(grades):.2f}")
    print("\n")

    print("Average Grades per District per Year:")
    for district, years in district_year_avg_grades.items():
        for year, avg_grade in years.items():
            print(f"District: {district}, Year: {year}, Average Grade: {avg_grade:.2f}")
    print("\n")

filenames = ['2018nc.csv', '2019nc.csv', '2022nc.csv', '2023nc.csv']

results = analyze_scores(filenames)

plot_histograms(results)

plot_mean_scores_per_year(results)

plot_proportions_per_score(results)

plot_mean_grades_per_year(results)

plot_proportions_per_grade(results)

print_results(results)
