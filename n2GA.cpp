#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
#include <climits>


using namespace std;

struct Task {
    int machine;
    int time;
};

int n,m,chromlen;
int populationsize;
int numReruns;
int numIter;
float mutationProb;
int mutationType;
int crossoverType;

random_device rd;
mt19937 gen(rd());

uniform_real_distribution<float> fdis(0.0f,1.0f);

inline int randint(int limit){
    return uniform_int_distribution<>(0,limit-1)(gen);
}

inline int fitness(
    const vector<int>& chromosome,
    const vector<vector<Task>>& jobs
){

    vector<int> nextTaskPtr(m,0);
    vector<int> nextTaskTime(m,0);
    vector<int> nextFreeSlot(n,0);

    for(int i=0;i<chromlen;i++){

        int curJob=chromosome[i];

        const Task& curTask=
            jobs[curJob]
                [nextTaskPtr[curJob]++];

        int machine=curTask.machine;
        int load=curTask.time;

        int start=max(
            nextFreeSlot[machine],
            nextTaskTime[curJob]
        );

        int finish=start+load;

        nextFreeSlot[machine]=finish;
        nextTaskTime[curJob]=finish;
    }

    return *max_element(
        nextFreeSlot.begin(),
        nextFreeSlot.end()
    );
}

vector<int> interleavingcrossover(
    const vector<int>& A,
    const vector<int>& B
){

    vector<int> child(chromlen);
    vector<int> used(m,0);

    int ap=0;
    int bp=0;

    for(int i=0;i<chromlen;i++){

        if(i&1){

            while(
                used[B[bp]]>=n
            ) bp++;

            child[i]=B[bp++];
        }
        else{

            while(
                used[A[ap]]>=n
            ) ap++;

            child[i]=A[ap++];
        }

        used[child[i]]++;
    }

    return child;
}

vector<int> segmentcrossover(
    const vector<int>& A,
    const vector<int>& B
){

    int l=randint(chromlen);
    int r=randint(chromlen);

    if(l>r) swap(l,r);

    vector<int> child(chromlen,-1);

    vector<int> count(m,0);

    for(int i=l;i<=r;i++){

        child[i]=A[i];
        count[A[i]]++;
    }

    int bp=0;

    for(int i=0;i<chromlen;i++){

        if(child[i]!=-1)
            continue;

        while(
            count[B[bp]]>=n
        ){
            bp++;
        }

        child[i]=B[bp];
        count[B[bp]]++;
        bp++;
    }

    return child;
}

inline vector<int> crossover(
    const vector<int>& A,
    const vector<int>& B
){

    if(crossoverType==1)
        return interleavingcrossover(A,B);

    return segmentcrossover(A,B);
}

void swapmutation(
    vector<int>& c
){

    int a=randint(chromlen);
    int b=randint(chromlen);

    swap(c[a],c[b]);
}

void insertmutation(
    vector<int>& c
){

    int a=randint(chromlen);
    int b=randint(chromlen);

    if(a==b)
        return;

    int val=c[a];

    if(a<b){

        move(
            c.begin()+a+1,
            c.begin()+b+1,
            c.begin()+a
        );
    }
    else{

        move_backward(
            c.begin()+b,
            c.begin()+a,
            c.begin()+a+1
        );
    }

    c[b]=val;
}

void invertmutation(
    vector<int>& c
){

    int a=randint(chromlen);
    int b=randint(chromlen);

    if(a>b)
        swap(a,b);

    reverse(
        c.begin()+a,
        c.begin()+b+1
    );
}

inline void mutation(
    vector<int>& c
){

    if(mutationType==1)
        swapmutation(c);

    else if(mutationType==2)
        insertmutation(c);

    else
        invertmutation(c);
}

inline int tournament(
    const vector<int>& fits
){

    int a=randint(populationsize);
    int b=randint(populationsize);

    return (
        fits[a]<fits[b]
    )?a:b;
}

void local_search(
    vector<int>& chrom,
    const vector<vector<Task>>& jobs
){

    int best=
        fitness(chrom,jobs);

    for(int k=0;k<8;k++){

        int a=randint(chromlen);
        int b=randint(chromlen);

        swap(
            chrom[a],
            chrom[b]
        );

        int f=
            fitness(chrom,jobs);

        if(f<best){

            best=f;
        }
        else{

            swap(
                chrom[a],
                chrom[b]
            );
        }
    }
}

int main(){

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin>>populationsize;
    cin>>numReruns;
    cin>>numIter;
    cin>>mutationProb;
    cin>>mutationType;
    cin>>crossoverType;

    int t;
    cin>>t;

    while(t--){

        cin>>m>>n;

        chromlen=n*m;

        vector<vector<Task>>
            jobs(
                m,
                vector<Task>(n)
            );

        for(int i=0;i<m;i++){

            for(int j=0;j<n;j++){

                cin>>
                jobs[i][j].machine>>
                jobs[i][j].time;
            }
        }

        int optimum=INT_MAX;

        vector<vector<int>>
            population(
                populationsize,
                vector<int>(chromlen)
            );

        vector<vector<int>>
            nextPop(
                populationsize,
                vector<int>(chromlen)
            );

        for(int rerun=0;
            rerun<numReruns;
            rerun++){

            for(auto& chrom:
                population){

                iota(
                    chrom.begin(),
                    chrom.end(),
                    0
                );

                shuffle(
                    chrom.begin(),
                    chrom.end(),
                    gen
                );

                for(
                    auto& x:
                    chrom
                ){

                    x%=m;
                }
            }

            for(
                int iter=0;
                iter<numIter;
                iter++
            ){

                vector<int>
                    fits(
                        populationsize
                    );

                for(
                    int i=0;
                    i<populationsize;
                    i++
                ){

                    fits[i]=
                        fitness(
                            population[i],
                            jobs
                        );
                }

                vector<int>
                    order(
                        populationsize
                    );

                iota(
                    order.begin(),
                    order.end(),
                    0
                );

                sort(
                    order.begin(),
                    order.end(),
                    [&](int a,int b){

                    return
                        fits[a]
                        <
                        fits[b];

                });

                optimum=min(
                    optimum,
                    fits[
                        order[0]
                    ]
                );

                nextPop[0]=
                    population[
                        order[0]
                    ];

                nextPop[1]=
                    population[
                        order[1]
                    ];

                for(
                    int i=2;
                    i<populationsize;
                    i+=2
                ){

                    int p1=
                        tournament(
                            fits
                        );

                    int p2=
                        tournament(
                            fits
                        );

                    nextPop[i]=
                        crossover(
                            population[p1],
                            population[p2]
                        );

                    if(i+1<
                       populationsize){

                        nextPop[i+1]=
                            crossover(
                                population[p2],
                                population[p1]
                            );
                    }

                    for(
                        int c=i;
                        c<
                        min(
                        i+2,
                        populationsize
                        );
                        c++
                    ){

                        if(
                        fdis(gen)
                        <
                        mutationProb
                        ){

                            mutation(
                                nextPop[c]
                            );
                        }

                        local_search(
                            nextPop[c],
                            jobs
                        );
                    }
                }

                population.swap(
                    nextPop
                );
            }
        }

        cout
        <<
        optimum
        <<
        '\n';
    }
}
