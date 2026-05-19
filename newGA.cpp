#include <bits/stdc++.h>
using namespace std;

struct Task {
    int machine;
    int time;
};

static int n, m, chromlen;
static int populationsize, numReruns, numIter;
static float mutationProb;
static int mutationType, crossoverType;

static mt19937 gen(123456);
static uniform_real_distribution<float> fdis(0.0f, 1.0f);
static uniform_int_distribution<int> rdist;

// ---------------------------
// FITNESS (cache-friendly)
// ---------------------------
int fitness(const vector<int> &chromosome,
            const vector<vector<Task>> &jobs)
{
    vector<int> jobPtr(m, 0);
    vector<int> jobReady(m, 0);
    vector<int> machineReady(n, 0);

    for (int i = 0; i < chromlen; i++)
    {
        int job = chromosome[i];
        const Task &t = jobs[job][jobPtr[job]++];

        int start = max(jobReady[job], machineReady[t.machine]);
        int end = start + t.time;

        jobReady[job] = end;
        machineReady[t.machine] = end;
    }

    return *max_element(machineReady.begin(), machineReady.end());
}

// ---------------------------
// CROSSOVER 1: INTERLEAVING
// ---------------------------
vector<int> interleaving(const vector<int> &a, const vector<int> &b)
{
    vector<int> child(chromlen);
    vector<int> used(m, 0);

    int ap = 0, bp = 0;

    for (int i = 0; i < chromlen; i++)
    {
        if (i % 2 == 0)
        {
            while (used[a[ap]] >= n) ap++;
            child[i] = a[ap++];
        }
        else
        {
            while (used[b[bp]] >= n) bp++;
            child[i] = b[bp++];
        }
        used[child[i]]++;
    }
    return child;
}

// ---------------------------
// CROSSOVER 2: SEGMENT + REPAIR
// ---------------------------
vector<int> segment(const vector<int> &a, const vector<int> &b)
{
    int i1 = rdist(gen) % chromlen;
    int i2 = rdist(gen) % chromlen;
    if (i1 > i2) swap(i1, i2);

    vector<int> child(chromlen);
    vector<int> count(m, 0);

    auto take = [&](int val, int &bad) {
        if (count[val] < n) {
            count[val]++;
            return val;
        }
        bad++;
        return -1;
    };

    int bad = 0;

    for (int i = 0; i < i1; i++)
        child[i] = take(a[i], bad);

    for (int i = i1; i <= i2; i++)
        child[i] = take(b[i], bad);

    for (int i = i2 + 1; i < chromlen; i++)
        child[i] = take(a[i], bad);

    vector<int> missing;
    missing.reserve(bad);

    for (int j = 0; j < m; j++)
        for (int k = count[j]; k < n; k++)
            missing.push_back(j);

    shuffle(missing.begin(), missing.end(), gen);

    int p = 0;
    for (int &x : child)
        if (x == -1) x = missing[p++];

    return child;
}

// ---------------------------
vector<int> crossover(const vector<int> &a,
                      const vector<int> &b,
                      int type)
{
    if (type == 1) return interleaving(a, b);
    return segment(a, b);
}

// ---------------------------
// MUTATIONS
// ---------------------------
void swap_mut(vector<int> &c)
{
    int a = rdist(gen) % chromlen;
    int b = rdist(gen) % chromlen;
    swap(c[a], c[b]);
}

void insert_mut(vector<int> &c)
{
    int a = rdist(gen) % chromlen;
    int b = rdist(gen) % chromlen;
    if (a == b) return;
    if (a > b) swap(a, b);

    int tmp = c[a];
    for (int i = a; i < b; i++)
        c[i] = c[i + 1];
    c[b] = tmp;
}

void invert_mut(vector<int> &c)
{
    int a = rdist(gen) % chromlen;
    int b = rdist(gen) % chromlen;
    if (a > b) swap(a, b);

    while (a < b)
        swap(c[a++], c[b--]);
}

void mutation(vector<int> &c, int type)
{
    if (type == 1) swap_mut(c);
    else if (type == 2) insert_mut(c);
    else invert_mut(c);
}

// ---------------------------
// TOURNAMENT SELECTION
// ---------------------------
int tournament(const vector<int> &fit)
{
    int a = rdist(gen) % populationsize;
    int b = rdist(gen) % populationsize;
    return (fit[a] < fit[b]) ? a : b;
}

// ---------------------------
// MAIN
// ---------------------------
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> populationsize >> numReruns >> numIter
        >> mutationProb >> mutationType >> crossoverType;

    int T;
    cin >> T;

    while (T--)
    {
        cin >> m >> n;
        chromlen = n * m;

        vector<vector<Task>> jobs(m, vector<Task>(n));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                cin >> jobs[i][j].machine >> jobs[i][j].time;

        rdist = uniform_int_distribution<int>(0, chromlen - 1);

        vector<vector<int>> pop(populationsize, vector<int>(chromlen));
        vector<vector<int>> next(populationsize, vector<int>(chromlen));
        vector<int> fit(populationsize);

        int best = INT_MAX;

        for (int r = 0; r < numReruns; r++)
        {
            for (auto &p : pop)
                for (int &x : p)
                    x = gen() % m;

            for (int it = 0; it < numIter; it++)
            {
                // fitness
                for (int i = 0; i < populationsize; i++)
                    fit[i] = fitness(pop[i], jobs);

                int bestIdx = min_element(fit.begin(), fit.end()) - fit.begin();
                best = min(best, fit[bestIdx]);

                // elitism
                next[0] = pop[bestIdx];

                for (int i = 1; i < populationsize; i += 2)
                {
                    int p1 = tournament(fit);
                    int p2 = tournament(fit);

                    next[i] = crossover(pop[p1], pop[p2], crossoverType);
                    if (i + 1 < populationsize)
                        next[i + 1] = crossover(pop[p2], pop[p1], crossoverType);

                    if (fdis(gen) < mutationProb)
                        mutation(next[i], mutationType);

                    if (i + 1 < populationsize &&
                        fdis(gen) < mutationProb)
                        mutation(next[i + 1], mutationType);
                }

                pop.swap(next);
            }
        }

        cout << best << "\n";
    }

    return 0;
}
