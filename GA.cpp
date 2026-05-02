#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>

using namespace std;

typedef struct Task {
    int machine;
    int time;
} task;

int n; // number of machines
int m; // number of jobs
int chromlen;
int populationsize;
int numReruns;  // number of restarts
int numIter; // number of iterations
float mutationProb; // probability of mutation
int mutationType;
int crossoverType;

// RNG stuff
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dis(0, 1 << 20);
uniform_real_distribution<> fdis(0.0f, 1.0f);

int fitness(vector<int> &chromosome, vector<vector<task>> jobs) {
    vector<int> nextTaskPtr(m, 0);
    vector<int> nextTaskTime(m, 0);
    vector<int> nextFreeSlot(n, 0);
    
    for (int i = 0; i < n*m; ++i) {
        int cur_job = chromosome[i];
        task cur_task = jobs[cur_job][nextTaskPtr[cur_job]];
        nextTaskPtr[cur_job]++;
        int machine = cur_task.machine;
        int load = cur_task.time;
        int starttime = max(nextFreeSlot[machine], nextTaskTime[cur_job]);
        nextFreeSlot[machine] = starttime + load;
        nextTaskTime[cur_job] = starttime + load;
    }
    
    int makespan = 0;
    for (int i = 0; i < n; ++i) {
        makespan = max(nextFreeSlot[i], makespan);
    }
    return makespan;
}

vector<int> interleavingcrossover(vector<int> &parent_a, vector<int> &parent_b) {
    int chromlen = n*m;
    vector<int> tasksDone(m, 0);
    vector<int> child(chromlen);
    int aptr = 0;
    int bptr = 0;
    for (int i = 0; i < chromlen; ++i) {
        if (i % 2 == 0) {
            while (tasksDone[parent_a[aptr]] == n) {
                aptr++;
            }
            child[i] = parent_a[aptr];
            tasksDone[child[i]]++;
            aptr++;
        } else {
            while (tasksDone[parent_b[bptr]] == n) {
                bptr++;
            }
            child[i] = parent_b[bptr];
            tasksDone[child[i]]++;
            bptr++;
        }
    }
    return child;
}

vector<int> segmentcrossover(vector<int> &parent_a, vector<int> &parent_b) {
    int ind1 = dis(gen) % chromlen;
    int ind2 = dis(gen) % chromlen;
    while (ind1 == ind2) {
        ind2 = dis(gen) % chromlen;
    }
    if (ind2 < ind1) {
        swap(ind1, ind2);
    }

    vector<int> child(chromlen);
    vector<int> tasksDone(m, 0);
    int toRepair = 0;

    for (int i = 0; i < ind1; ++i) {
        child[i] = parent_a[i];
        if (tasksDone[child[i]] < n) {
            tasksDone[child[i]]++;
        } else {
            toRepair++;
            child[i] = -1;
        }
    }
    for (int i = ind1; i <= ind2; ++i) {
        child[i] = parent_b[i];
        if (tasksDone[child[i]] < n) {
            tasksDone[child[i]]++;
        } else {
            toRepair++;
            child[i] = -1;
        }
    }
    for (int i = ind2 + 1; i < chromlen; ++i) {
        child[i] = parent_a[i];
        if (tasksDone[child[i]] < n) {
            tasksDone[child[i]]++;
        } else {
            toRepair++;
            child[i] = -1;
        }
    }

    if (toRepair == 0) {
        return child;
    }
    
    vector<int> missing(toRepair);
    int missingptr = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = tasksDone[i]; j < n; ++j) {
            missing[missingptr] = i;
            missingptr++;
        }
    }

    shuffle(missing.begin(), missing.end(), gen);
    int repairptr = 0;

    for (int i = 0; i < chromlen; ++i) {
        if (child[i] == -1) {
            child[i] = missing[repairptr];
            repairptr++;
        }
    }
    return child;
}

vector<int> crossover(vector<int> &parent_a, vector<int> &parent_b, int type) {
    if (type == 1) {
        return interleavingcrossover(parent_a, parent_b);
    } else {
        return segmentcrossover(parent_a, parent_b);
    }
}

void swapmutation(vector<int> &chromosome) {
    int ind1 = dis(gen) % chromlen;
    int ind2 = dis(gen) % chromlen;
    while (ind1 == ind2) {
        ind2 = dis(gen) % chromlen;
    }
    int temp = chromosome[ind1];
    chromosome[ind1] = chromosome[ind2];
    chromosome[ind2] = temp;
}

void insertmutation(vector<int> &chromosome) {
    int ind1 = dis(gen) % chromlen;
    int ind2 = dis(gen) % chromlen;
    while (ind1 == ind2) {
        ind2 = dis(gen) % chromlen;
    }
    int temp = chromosome[ind1];
    if (ind1 < ind2) {
        for (int i = ind1 + 1; i <= ind2; ++i) {
            chromosome[i-1] = chromosome[i];
        }
    } else {
        for (int i = ind1 - 1; i >= ind2; --i) {
            chromosome[i+1] = chromosome[i];
        }
    }
    chromosome[ind2] = temp;
}

void invertmutation(vector<int> &chromosome) {
    int ind1 = dis(gen) % chromlen;
    int ind2 = dis(gen) % chromlen;
    while (ind1 == ind2) {
        ind2 = dis(gen) % chromlen;
    }
    if (ind1 > ind2) {
        swap(ind1, ind2);
    }
    while (ind1 < ind2) {
        swap(chromosome[ind1], chromosome[ind2]);
        ind1++;
        ind2--;
    }
}

void mutation(vector<int> &chromosome, int type) {
    if (type == 1) {
        swapmutation(chromosome);
    } else if (type == 2) {
        insertmutation(chromosome);
    } else {
        invertmutation(chromosome);
    }
}

int main() {
    cin >> populationsize;
    cin >> numReruns;
    cin >> numIter;
    cin >> mutationProb;
    cin >> mutationType;
    cin >> crossoverType;
    int t;
    cin >> t;
    while (t--) {
        cin >> m;
        cin >> n;
        chromlen = n*m;
        vector<vector<task>> jobs(m, vector<task>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cin >> jobs[i][j].machine;
                cin >> jobs[i][j].time;
            }
        }
        int optimum = 1 << 30;
        vector<vector<int>> population(populationsize, vector<int>(chromlen));
        vector<vector<int>> nextPopulation(populationsize, vector<int>(chromlen));
        for (int rerun = 0; rerun < numReruns; ++rerun) {
            for (int i = 0; i < populationsize; ++i) {
                iota(population[i].begin(), population[i].end(), 0);
                shuffle(population[i].begin(), population[i].end(), gen);
                for (int j = 0; j < chromlen; ++j) {
                    population[i][j] %= m;
                }
            }
            for (int iteration = 0; iteration < numIter; ++iteration) {
                vector<pair<int, vector<int>>> pop_with_fit;
                for (auto &ind : population) {
                    pop_with_fit.push_back({fitness(ind, jobs), ind});
                }
                sort(pop_with_fit.begin(), pop_with_fit.end());
                for (int i = 0; i < populationsize; ++i) {
                    population[i] = pop_with_fit[i].second;
                }
                //cout << "current generation's best fit: " << (fitness(population.front(), jobs)) << '\n';
                optimum = min(fitness(population.front(), jobs), optimum);
                for (int i = 0; i < populationsize; i += 2) {
                    nextPopulation[i] = crossover(population[i], population[i+1], crossoverType);
                    nextPopulation[i+1] = crossover(population[i+1], population[i], crossoverType);
                    if (fdis(gen) <= mutationProb) {
                        mutation(nextPopulation[i], mutationType);
                    }
                    if (fdis(gen) <= mutationProb) {
                        mutation(nextPopulation[i+1], mutationType);
                    }
                }
                population.swap(nextPopulation);
                
            }
            //cout << "optimum after rerun " << rerun << ": "<< optimum << '\n';
        }
        cout << optimum << '\n';
    }
    

    return 0;
}