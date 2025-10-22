// Portable, CPU-only, multithreaded dir deduper (size+CRC32)
// Build: g++ -O3 -march=native -pthread -std=c++17 dirhash_fast_mt.cpp -o dirhash_fast_mt
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_acle.h>
#endif

using namespace std;

// ---------------- CRC32 (IEEE) ----------------
static uint32_t CRC32_TAB[256];
static void crc32_init() {
    uint32_t poly = 0xEDB88320u;
    for (uint32_t i=0;i<256;i++){
        uint32_t c=i;
        for (int j=0;j<8;j++)
            c = (c&1)? (poly ^ (c>>1)) : (c>>1);
        CRC32_TAB[i]=c;
    }
}
static inline uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n){
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    size_t i = 0;
    // 64-bit chunks
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        // use memcpy to avoid alignment issues
        memcpy(&v, p + i, 8);
        crc = __crc32cd(crc, v);
    }
    // remaining bytes
    for (; i < n; ++i) {
        crc = __crc32cb(crc, p[i]);
    }
    return crc;
#else
    for (size_t i=0;i<n;i++) crc = CRC32_TAB[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
#endif
}

// ---------------- Timer ----------------
static inline double now_ms(){
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------- CLI ----------------
struct Args {
    string root;
    uint64_t min_bytes = 0;
    int top = 10;
    int threads = 0;
};
static Args parse(int argc, char** argv){
    Args a;
    if (argc < 2){ fprintf(stderr,"usage: %s ROOT [--min-bytes N] [--top K] [--threads T]\n", argv[0]); exit(2); }
    a.root = argv[1];
    for (int i=2;i<argc;i++){
        string s=argv[i];
        if (s=="--min-bytes" && i+1<argc) a.min_bytes=stoull(argv[++i]);
        else if (s=="--top" && i+1<argc) a.top=stoi(argv[++i]);
        else if (s=="--threads" && i+1<argc) a.threads=stoi(argv[++i]);
    }
    if (a.threads<=0){ int n=std::thread::hardware_concurrency(); a.threads = n>0?n:4; }
    return a;
}

// ---------------- File walk ----------------
struct FileEnt { string path; uint64_t size; };

static bool is_regular(const string& p, uint64_t& sz){
    struct stat st{};
    if (lstat(p.c_str(), &st)!=0) return false;
    if (!S_ISREG(st.st_mode)) return false;
    sz = (uint64_t)st.st_size;
    return true;
}
static void walk_dir(const string& root, vector<FileEnt>& out){
    vector<string> st; st.push_back(root);
    while(!st.empty()){
        string d=move(st.back()); st.pop_back();
        DIR* dir=opendir(d.c_str());
        if(!dir) continue;
        while(auto* ent=readdir(dir)){
            const char* name=ent->d_name;
            if (strcmp(name,".")==0 || strcmp(name,"..")==0) continue;
            string p = d + "/" + name;
            struct stat stt{};
            if (lstat(p.c_str(), &stt)!=0) continue;
            if (S_ISDIR(stt.st_mode)) st.push_back(p);
            else if (S_ISREG(stt.st_mode)) out.push_back({p, (uint64_t)stt.st_size});
        }
        closedir(dir);
    }
}

// ---------------- Key + hashing ----------------
struct Key {
    uint64_t size;
    uint32_t crc;
    bool operator==(const Key& o) const { return size==o.size && crc==o.crc; }
};
struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        uint64_t x = k.size ^ (uint64_t(k.crc) << 1);
        x ^= (x >> 33); x *= 0xff51afd7ed558ccdULL;
        x ^= (x >> 33); x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= (x >> 33);
        return (size_t)x;
    }
};

// ---------------- Hash one file ----------------
static Key hash_file(const string& path, uint64_t sz){
    int fd = open(path.c_str(), O_RDONLY);
    if (fd<0) return {sz, 0};
    const size_t BUF=4<<20; // 4 MiB
#ifdef POSIX_FADV_SEQUENTIAL
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    static thread_local vector<uint8_t> buf(BUF);
    uint32_t crc=0xFFFFFFFFu;
    while(true){
        ssize_t n = read(fd, buf.data(), BUF);
        if (n<0){ crc=0; break; }
        if (n==0) break;
        crc = crc32_update(crc, buf.data(), (size_t)n);
    }
    close(fd);
    crc ^= 0xFFFFFFFFu;
    return {sz, crc};
}

// ---------------- Main ----------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    crc32_init();
    Args args = parse(argc, argv);
    const double t0 = now_ms();

    vector<FileEnt> files;
    files.reserve(1<<16);
    walk_dir(args.root, files);

    vector<FileEnt> todo; todo.reserve(files.size());
    uint64_t bytes_total=0;
    for (auto& f: files){
        if (f.size >= args.min_bytes){ todo.push_back(f); bytes_total += f.size; }
    }

    atomic<size_t> idx{0};
    const int T = args.threads;
    vector<unordered_map<Key, vector<string>, KeyHash>> locals(T);
    for (int t=0;t<T;t++){ locals[t].reserve(4096); }

    auto worker = [&](int tid){
        size_t i;
        while ( (i = idx.fetch_add(1, memory_order_relaxed)) < todo.size() ){
            const auto& fe = todo[i];
            Key k = hash_file(fe.path, fe.size);
            auto& map = locals[tid];
            auto it = map.find(k);
            if (it==map.end()){
                map.emplace(k, vector<string>{fe.path});
            } else {
                it->second.emplace_back(fe.path);
            }
        }
    };

    vector<thread> pool; pool.reserve(T);
    for (int t=0;t<T;t++) pool.emplace_back(worker, t);
    for (auto& th: pool) th.join();

    unordered_map<Key, vector<string>, KeyHash> groups; groups.reserve(todo.size()/2+1);
    for (int t=0;t<T;t++){
        for (auto& kv : locals[t]){
            auto it = groups.find(kv.first);
            if (it==groups.end()) groups.emplace(kv.first, move(kv.second));
            else {
                auto& v = it->second; auto& w = kv.second;
                v.insert(v.end(), make_move_iterator(w.begin()), make_move_iterator(w.end()));
            }
        }
    }

    struct G { Key k; size_t cnt; uint64_t bytes; vector<string> samples; };
    vector<G> gs; gs.reserve(groups.size());
    for (auto& kv : groups){
        if (kv.second.size() >= 2){
            G g; g.k = kv.first; g.cnt = kv.second.size(); g.bytes = kv.first.size * g.cnt;
            for (size_t i=0;i<kv.second.size() && i<3;i++) g.samples.push_back(move(kv.second[i]));
            gs.push_back(move(g));
        }
    }
    sort(gs.begin(), gs.end(), [](const G& a, const G& b){
        if (a.bytes!=b.bytes) return a.bytes>b.bytes;
        if (a.cnt!=b.cnt) return a.cnt>b.cnt;
        if (a.k.size!=b.k.size) return a.k.size>b.k.size;
        return a.k.crc<b.k.crc;
    });
    if ((int)gs.size() > args.top) gs.resize(args.top);

    const double t1 = now_ms();

    cout << "{"
         << "\"ok\":true,"
         << "\"files_scanned\":" << files.size() << ","
         << "\"bytes_scanned\":" << bytes_total << ","
         << "\"groups\":[";
    for (size_t i=0;i<gs.size();i++){
        const auto& g=gs[i];
        char keybuf[64];
        snprintf(keybuf, sizeof(keybuf), "sz=%llu:crc32=%08X",
                 (unsigned long long)g.k.size, (unsigned)g.k.crc);
        cout << "{\"key\":\"" << keybuf << "\",";
        cout << "\"count\":" << g.cnt << ",";
        cout << "\"total_bytes\":" << g.bytes << ",";
        cout << "\"samples\":[";
        for (size_t j=0;j<g.samples.size();j++){
            string s = g.samples[j];
            for (auto& ch : s) if (ch=='"') ch='\'';
            cout << "\"" << s << "\"";
            if (j+1<g.samples.size()) cout << ",";
        }
        cout << "]}";
        if (i+1<gs.size()) cout << ",";
    }
    cout << "],"
         << "\"top\":" << args.top << ","
         << "\"threads\":" << T << ","
         << fixed << setprecision(1)
         << "\"ms\":" << (t1 - t0)
         << "}\n";
    return 0;
}


