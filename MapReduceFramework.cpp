#include "MapReduceFramework.h"
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include "algorithm"
#include <map>
#include "Barrier.h"

#define SYS_ERR "system error: "
#define P_THREAD_FAILED "failed\n"

//typedef struct {
//    stage_t stage;
//    float percentage;
//} JobState;
struct JobContext;

struct ThreadContext {
    JobContext* job_context;
    int serial_num;
    pthread_t thread_ID;
    IntermediateVec intermediate_vec;
    ThreadContext(): job_context(nullptr){}
    ThreadContext(JobContext* job_context_, int serial_num_):
            job_context(job_context_), serial_num(serial_num_), thread_ID(0){}
};

struct JobContext {
    ThreadContext* threads;
    int multi_thread_level;
    std::atomic<uint64_t>* atomic_counter;
    const InputVec inputVec;
    OutputVec* outputVec;
    const MapReduceClient* client;
    Barrier barrier;
    bool is_waiting = false;
    std::vector<IntermediateVec> shuffled_vec;
    pthread_mutex_t emit3_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t wait_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t atomic_counter_mutex = PTHREAD_MUTEX_INITIALIZER;
    JobContext(ThreadContext* threadContexts, int multiThreadLevel, std::atomic<uint64_t>* atomicCounter,
               const InputVec* input, OutputVec* output, const MapReduceClient* mapReduceClient,
               const Barrier& barrier)
            : threads(threadContexts), multi_thread_level(multiThreadLevel), atomic_counter(atomicCounter),
              inputVec(*input), outputVec(output), client(mapReduceClient), barrier(barrier) {}
};


void emit2 (K2* key, V2* value, void* context){
    auto tc = (ThreadContext*)(context);
    IntermediatePair mapped_cur = {key, value};
    tc->intermediate_vec.push_back(mapped_cur);
}

void emit3 (K3* key, V3* vaJobContextlue, void* context){
    auto jc = (JobContext*)(context);
    pthread_mutex_lock(&jc->emit3_mutex);
    jc->outputVec->push_back({key, vaJobContextlue});
    pthread_mutex_unlock(&jc->emit3_mutex);
}

void map_phase(ThreadContext* thread_context){
    while (true){
        pthread_mutex_lock(&thread_context->job_context->atomic_counter_mutex);
        uint64_t cur_idx = (++(*thread_context->job_context->atomic_counter)&(0x7FFFFFFF));
        if (cur_idx > (thread_context->job_context->inputVec.size())){
            --(*thread_context->job_context->atomic_counter);
            pthread_mutex_unlock(&thread_context->job_context->atomic_counter_mutex);
            break;
        }
        pthread_mutex_unlock(&thread_context->job_context->atomic_counter_mutex);
        auto cur = thread_context->job_context->inputVec[(int)(cur_idx)-1];
        thread_context->job_context->client->map(cur.first, cur.second, thread_context);
    }
}

struct Comparator {
    bool operator()(const K2* key1, const K2* key2) const {
        return *key1<*key2;
    }
};

void shuffle_phase(JobContext* job_context){
    pthread_mutex_lock(&job_context->atomic_counter_mutex);
    job_context->atomic_counter->store(0);
    *job_context->atomic_counter += (uint64_t)(1) << 63;
    *job_context->atomic_counter += ((uint64_t)(job_context->multi_thread_level)) << 31;
    pthread_mutex_unlock(&job_context->atomic_counter_mutex);
    std::map<K2*, std::vector<std::pair<K2*, V2*>>, Comparator> shuffled_map;
    for (int i = 0; i < job_context->multi_thread_level; i++) {
        ++(*job_context->atomic_counter);
        for (auto pair:(job_context->threads + i)->intermediate_vec) {
            auto it = shuffled_map.find((pair.first));
            if (it == shuffled_map.end()){
                std::vector<std::pair<K2*, V2*>> values_vec = {pair};
                shuffled_map.insert({pair.first, values_vec});
            } else{
                it->second.push_back(pair);
            }
        }
    }
    for (auto it = shuffled_map.rbegin(); it!=shuffled_map.rend(); it++){
        IntermediateVec intermediate_vec;
        for (auto val: it->second) {
            intermediate_vec.push_back({val});
        }
        job_context->shuffled_vec.push_back(intermediate_vec);
    }
    pthread_mutex_lock(&job_context->atomic_counter_mutex);
    job_context->atomic_counter->store(0);
    *job_context->atomic_counter += ((uint64_t)(3)<<62) + ((uint64_t)(job_context->shuffled_vec.size())<<31);
    pthread_mutex_unlock(&job_context->atomic_counter_mutex);
}

void reduce_phase(JobContext* job_context){
    while (true){
        pthread_mutex_lock(&job_context->atomic_counter_mutex);
        uint64_t cur_idx = ((++(*job_context->atomic_counter))&(0x7FFFFFFF));
        if(cur_idx > job_context->shuffled_vec.size()){
            --(*job_context->atomic_counter);
            pthread_mutex_unlock(&job_context->atomic_counter_mutex);
            break;
        }
        pthread_mutex_unlock(&job_context->atomic_counter_mutex);
        job_context->client->reduce(&(job_context->shuffled_vec[cur_idx-1]), job_context);
    }
}

void* general_thread_run(void* thread_context){
    auto tc = (ThreadContext*)thread_context;
    map_phase(tc);
    tc->job_context->barrier.barrier();
    if (tc->serial_num == 0){
        shuffle_phase(tc->job_context);
    }
    tc->job_context->barrier.barrier();
    reduce_phase(tc->job_context);
    return nullptr;
}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
    auto atomic_counter = new std::atomic<uint64_t>(0);
    *atomic_counter += ((uint64_t )1 << 62);
    *atomic_counter += (inputVec.size() << 31);
    auto jobContext = new JobContext{new ThreadContext[multiThreadLevel], multiThreadLevel,
                                     atomic_counter, &inputVec, &outputVec, &client, Barrier(multiThreadLevel)};
    for (int t = 0; t < multiThreadLevel; t++) {
        jobContext->threads[t] = ThreadContext {jobContext, t};
        if (pthread_create(&(jobContext->threads[t].thread_ID), NULL,
                           &general_thread_run, (void*)(jobContext->threads+t)) != 0){
            std::cerr << SYS_ERR << P_THREAD_FAILED;
            exit(1);
        }
    }
    return (JobHandle) jobContext;
}

void waitForJob(JobHandle job){
    auto job_context = (JobContext*)job;
    pthread_mutex_lock(&job_context->wait_mutex);
    if (job_context->is_waiting){
        return;
    }
    job_context->is_waiting = true;
    pthread_mutex_unlock(&job_context->wait_mutex);
    for (int i = 0; i < job_context->multi_thread_level; i++) {
        pthread_join(job_context->threads[i].thread_ID, NULL);
    }
}

void getJobState(JobHandle job, JobState* state){
    auto job_context = (JobContext *) job;
    pthread_mutex_lock(&job_context->atomic_counter_mutex);
    auto partial = (float)((job_context->atomic_counter->load())&(0x7FFFFFFF));
    auto total = (float)(((job_context->atomic_counter->load())>>31)&(0x7FFFFFFF));
    state->stage = static_cast<stage_t>(((job_context->atomic_counter->load())&((uint64_t)(3)<<62))>>62);
    pthread_mutex_unlock(&job_context->atomic_counter_mutex);
    state->percentage = (partial/total)*100;
}

void closeJobHandle(JobHandle job){
    waitForJob(job);
    auto job_context = (JobContext *) job;
    delete job_context->atomic_counter;
    delete[] job_context->threads;
    delete job_context;
}


