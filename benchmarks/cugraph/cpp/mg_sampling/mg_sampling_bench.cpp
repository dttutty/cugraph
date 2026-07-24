/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two-rank multi-GPU neighbor-sampling benchmark. It intentionally uses the
 * public libcugraph C API so the same source can be linked against a baseline
 * or candidate cuGraph build.
 */

#include <cugraph/partition_manager.hpp>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/random.h>
#include <cugraph_c/sampling_algorithms.h>

#include <raft/comms/std_comms.hpp>
#include <raft/core/handle.hpp>

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
  int rank{0};
  int world_size{2};
  int device{0};
  std::string master_addr{"127.0.0.1"};
  int port{39001};
  int32_t vertices{200000};
  int32_t degree{16};
  int32_t seeds_per_rank{512};
  std::vector<int32_t> fanout{15, 15};
  int warmup{1};
  int iters{3};
  bool renumber{true};
  bool return_hops{true};
  bool label_offsets{true};
  bool edge_ids{false};
  bool with_replacement{false};
  bool build_only{false};
};

[[noreturn]] void die(std::string const& message)
{
  std::cerr << "ERROR: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << operation << ": " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
    die(out.str());
  }
}

void check_nccl(ncclResult_t status, char const* operation)
{
  if (status != ncclSuccess) {
    std::ostringstream out;
    out << operation << ": " << ncclGetErrorString(status);
    die(out.str());
  }
}

void check_cugraph(cugraph_error_code_t status, cugraph_error_t* error, char const* operation)
{
  if (status != CUGRAPH_SUCCESS) {
    std::ostringstream out;
    out << operation;
    if (error != nullptr) { out << ": " << cugraph_error_message(error); }
    die(out.str());
  }
}

int parse_int(std::string const& text, char const* option)
{
  char* end{nullptr};
  errno = 0;
  auto value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
    die(std::string{"invalid integer for "} + option + ": " + text);
  }
  return static_cast<int>(value);
}

bool parse_bool(std::string const& text, char const* option)
{
  if (text == "1" || text == "true" || text == "yes") { return true; }
  if (text == "0" || text == "false" || text == "no") { return false; }
  die(std::string{"invalid boolean for "} + option + ": " + text);
}

std::vector<int32_t> parse_fanout(std::string const& text)
{
  std::vector<int32_t> fanout;
  std::stringstream stream{text};
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) { die("--fanout contains an empty value"); }
    auto value = parse_int(item, "--fanout");
    if (value == 0 || value < -1) { die("--fanout values must be -1 or positive"); }
    fanout.push_back(static_cast<int32_t>(value));
  }
  if (fanout.empty()) { die("--fanout must not be empty"); }
  return fanout;
}

Config parse_args(int argc, char** argv)
{
  Config config;
  for (int index = 1; index < argc; ++index) {
    std::string argument{argv[index]};
    auto equals = argument.find('=');
    auto key    = equals == std::string::npos ? argument : argument.substr(0, equals);
    auto value  = equals == std::string::npos ? std::string{} : argument.substr(equals + 1);

    auto require_value = [&]() {
      if (equals == std::string::npos) { die(key + " requires --key=value form"); }
    };

    if (key == "--rank") {
      require_value();
      config.rank = parse_int(value, "--rank");
    } else if (key == "--world-size") {
      require_value();
      config.world_size = parse_int(value, "--world-size");
    } else if (key == "--device") {
      require_value();
      config.device = parse_int(value, "--device");
    } else if (key == "--master-addr") {
      require_value();
      config.master_addr = value;
    } else if (key == "--port") {
      require_value();
      config.port = parse_int(value, "--port");
    } else if (key == "--vertices") {
      require_value();
      config.vertices = static_cast<int32_t>(parse_int(value, "--vertices"));
    } else if (key == "--degree") {
      require_value();
      config.degree = static_cast<int32_t>(parse_int(value, "--degree"));
    } else if (key == "--seeds-per-rank") {
      require_value();
      config.seeds_per_rank = static_cast<int32_t>(parse_int(value, "--seeds-per-rank"));
    } else if (key == "--fanout") {
      require_value();
      config.fanout = parse_fanout(value);
    } else if (key == "--warmup") {
      require_value();
      config.warmup = parse_int(value, "--warmup");
    } else if (key == "--iters") {
      require_value();
      config.iters = parse_int(value, "--iters");
    } else if (key == "--renumber") {
      require_value();
      config.renumber = parse_bool(value, "--renumber");
    } else if (key == "--return-hops") {
      require_value();
      config.return_hops = parse_bool(value, "--return-hops");
    } else if (key == "--label-offsets" || key == "--labels") {
      require_value();
      config.label_offsets = parse_bool(value, key.c_str());
    } else if (key == "--edge-ids") {
      require_value();
      config.edge_ids = parse_bool(value, "--edge-ids");
    } else if (key == "--with-replacement") {
      require_value();
      config.with_replacement = parse_bool(value, "--with-replacement");
    } else if (key == "--build-only") {
      config.build_only = true;
    } else if (key == "--help") {
      std::cout
        << "Usage: mg_sampling_bench --rank=N --world-size=2 --master-addr=HOST --port=P\n"
           "       [--vertices=N --degree=N --seeds-per-rank=N --fanout=15,15]\n"
           "       [--warmup=N --iters=N --edge-ids=true|false --renumber=true|false]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      die("unknown argument: " + argument);
    }
  }

  if (config.world_size != 2) { die("this benchmark currently requires --world-size=2"); }
  if (config.rank < 0 || config.rank >= config.world_size) {
    die("--rank must be in [0, world-size)");
  }
  if (config.device < 0) { die("--device must be non-negative"); }
  if (config.port < 1 || config.port > 65535) { die("--port must be in [1, 65535]"); }
  if (config.vertices <= 1 || config.degree <= 0 || config.seeds_per_rank < 0) {
    die("invalid graph sizing");
  }
  if (config.warmup < 0 || config.iters <= 0) { die("--warmup must be >= 0 and --iters > 0"); }
  auto edge_count = static_cast<int64_t>(config.vertices) * config.degree;
  if (config.edge_ids && edge_count > std::numeric_limits<int32_t>::max()) {
    die("vertices * degree exceeds INT32 edge-id range");
  }
  return config;
}

void write_exact(int descriptor, void const* data, size_t size)
{
  auto cursor = static_cast<char const*>(data);
  while (size > 0) {
    auto written = ::write(descriptor, cursor, size);
    if (written <= 0) { die("socket write failed"); }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
}

void read_exact(int descriptor, void* data, size_t size)
{
  auto cursor = static_cast<char*>(data);
  while (size > 0) {
    auto count = ::read(descriptor, cursor, size);
    if (count <= 0) { die("socket read failed"); }
    cursor += count;
    size -= static_cast<size_t>(count);
  }
}

int connect_with_retry(std::string const& host, int port)
{
  addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses{nullptr};
  auto service = std::to_string(port);
  if (::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses) != 0) {
    die("getaddrinfo failed for " + host);
  }

  for (int attempt = 0; attempt < 240; ++attempt) {
    for (auto* address = addresses; address != nullptr; address = address->ai_next) {
      int descriptor = ::socket(address->ai_family, address->ai_socktype, address->ai_protocol);
      if (descriptor < 0) { continue; }
      if (::connect(descriptor, address->ai_addr, address->ai_addrlen) == 0) {
        ::freeaddrinfo(addresses);
        return descriptor;
      }
      ::close(descriptor);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }

  ::freeaddrinfo(addresses);
  die("connect retry limit reached");
}

int accept_one(int port)
{
  addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags    = AI_PASSIVE;
  addrinfo* addresses{nullptr};
  auto service = std::to_string(port);
  if (::getaddrinfo(nullptr, service.c_str(), &hints, &addresses) != 0) {
    die("getaddrinfo failed for listen socket");
  }

  int listener{-1};
  for (auto* address = addresses; address != nullptr; address = address->ai_next) {
    listener = ::socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (listener < 0) { continue; }
    int enabled{1};
    static_cast<void>(::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)));
    if (::bind(listener, address->ai_addr, address->ai_addrlen) == 0 &&
        ::listen(listener, 1) == 0) {
      break;
    }
    ::close(listener);
    listener = -1;
  }
  ::freeaddrinfo(addresses);
  if (listener < 0) { die("failed to create listen socket"); }

  int connection = ::accept(listener, nullptr, nullptr);
  ::close(listener);
  if (connection < 0) { die("socket accept failed"); }
  return connection;
}

ncclUniqueId exchange_nccl_id(Config const& config)
{
  ncclUniqueId id{};
  if (config.rank == 0) {
    check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    int descriptor = accept_one(config.port);
    write_exact(descriptor, &id, sizeof(id));
    ::close(descriptor);
  } else {
    int descriptor = connect_with_retry(config.master_addr, config.port);
    read_exact(descriptor, &id, sizeof(id));
    ::close(descriptor);
  }
  return id;
}

uint64_t mix64(uint64_t value)
{
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

void make_local_edges(Config const& config,
                      std::vector<int32_t>& sources,
                      std::vector<int32_t>& destinations,
                      std::vector<int32_t>& edge_ids)
{
  auto vertices_per_rank = (config.vertices + config.world_size - 1) / config.world_size;
  auto first = std::min<int32_t>(config.rank * vertices_per_rank, config.vertices);
  auto last  = std::min<int32_t>(first + vertices_per_rank, config.vertices);
  auto local_vertices = std::max<int32_t>(last - first, 0);
  auto local_edges = static_cast<size_t>(local_vertices) * static_cast<size_t>(config.degree);
  sources.resize(local_edges);
  destinations.resize(local_edges);
  if (config.edge_ids) { edge_ids.resize(local_edges); }

  size_t index{0};
  for (int32_t offset = 0; offset < local_vertices; ++offset) {
    int32_t source = first + offset;
    for (int32_t neighbor = 0; neighbor < config.degree; ++neighbor) {
      int32_t destination = static_cast<int32_t>(
        mix64((static_cast<uint64_t>(source) << 32) ^
              static_cast<uint64_t>(neighbor + 0x9e3779b9U)) %
        static_cast<uint64_t>(config.vertices));
      if (destination == source) { destination = (destination + 1) % config.vertices; }
      sources[index]      = source;
      destinations[index] = destination;
      if (config.edge_ids) { edge_ids[index] = source * config.degree + neighbor; }
      ++index;
    }
  }
}

std::vector<int32_t> make_seeds(Config const& config)
{
  auto vertices_per_rank = (config.vertices + config.world_size - 1) / config.world_size;
  auto first = std::min<int32_t>(config.rank * vertices_per_rank, config.vertices);
  auto last  = std::min<int32_t>(first + vertices_per_rank, config.vertices);
  auto local_vertices = std::max<int32_t>(last - first, 1);
  std::vector<int32_t> seeds(static_cast<size_t>(config.seeds_per_rank));
  for (int32_t index = 0; index < config.seeds_per_rank; ++index) {
    seeds[static_cast<size_t>(index)] = first + ((index * 9973) % local_vertices);
  }
  return seeds;
}

struct DeviceArray {
  cugraph_type_erased_device_array_t* array{nullptr};
  cugraph_type_erased_device_array_view_t* view{nullptr};

  DeviceArray() = default;
  DeviceArray(DeviceArray const&) = delete;
  DeviceArray& operator=(DeviceArray const&) = delete;

  DeviceArray(DeviceArray&& other) noexcept : array(other.array), view(other.view)
  {
    other.array = nullptr;
    other.view  = nullptr;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept
  {
    if (this != &other) {
      reset();
      array       = other.array;
      view        = other.view;
      other.array = nullptr;
      other.view  = nullptr;
    }
    return *this;
  }

  void reset()
  {
    if (view != nullptr) {
      cugraph_type_erased_device_array_view_free(view);
      view = nullptr;
    }
    if (array != nullptr) {
      cugraph_type_erased_device_array_free(array);
      array = nullptr;
    }
  }

  ~DeviceArray() { reset(); }
};

DeviceArray make_device_array(cugraph_resource_handle_t const* handle,
                              void const* host_data,
                              size_t size,
                              cugraph_data_type_id_t type,
                              char const* name)
{
  DeviceArray output;
  cugraph_error_t* error{nullptr};
  auto status = cugraph_type_erased_device_array_create(handle, size, type, &output.array, &error);
  check_cugraph(status, error, (std::string{name} + " create").c_str());
  if (error != nullptr) {
    cugraph_error_free(error);
    error = nullptr;
  }
  output.view = cugraph_type_erased_device_array_view(output.array);
  status      = cugraph_type_erased_device_array_view_copy_from_host(
    handle, output.view, static_cast<byte_t const*>(host_data), &error);
  check_cugraph(status, error, (std::string{name} + " copy_from_host").c_str());
  if (error != nullptr) { cugraph_error_free(error); }
  return output;
}

void barrier_and_sync(raft::handle_t& handle)
{
  handle.get_comms().barrier();
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace

int main(int argc, char** argv)
{
  auto config = parse_args(argc, argv);
  check_cuda(cudaSetDevice(config.device), "cudaSetDevice");

  auto id = exchange_nccl_id(config);
  ncclComm_t nccl_communicator{};
  check_nccl(ncclCommInitRank(&nccl_communicator, config.world_size, id, config.rank),
             "ncclCommInitRank");

  raft::handle_t handle{};
  raft::comms::build_comms_nccl_only(
    &handle, nccl_communicator, config.world_size, config.rank);
  cugraph::partition_manager::init_subcomm(handle, 1);
  auto* c_handle = cugraph_create_resource_handle(&handle);
  if (c_handle == nullptr) { die("cugraph_create_resource_handle returned null"); }

  std::vector<int32_t> host_sources;
  std::vector<int32_t> host_destinations;
  std::vector<int32_t> host_edge_ids;
  make_local_edges(config, host_sources, host_destinations, host_edge_ids);
  auto host_seeds = make_seeds(config);
  std::vector<size_t> host_label_offsets{0, host_seeds.size()};

  auto device_sources = make_device_array(
    c_handle, host_sources.data(), host_sources.size(), INT32, "sources");
  auto device_destinations = make_device_array(
    c_handle, host_destinations.data(), host_destinations.size(), INT32, "destinations");
  DeviceArray device_edge_ids;
  if (config.edge_ids) {
    device_edge_ids = make_device_array(
      c_handle, host_edge_ids.data(), host_edge_ids.size(), INT32, "edge_ids");
  }
  auto device_seeds =
    make_device_array(c_handle, host_seeds.data(), host_seeds.size(), INT32, "seeds");
  DeviceArray device_label_offsets;
  if (config.label_offsets) {
    device_label_offsets = make_device_array(c_handle,
                                              host_label_offsets.data(),
                                              host_label_offsets.size(),
                                              SIZE_T,
                                              "label_offsets");
  }

  cugraph_error_t* error{nullptr};
  cugraph_graph_t* graph{nullptr};
  cugraph_graph_properties_t properties{FALSE, TRUE};
  cugraph_type_erased_device_array_view_t const* source_views[] = {device_sources.view};
  cugraph_type_erased_device_array_view_t const* destination_views[] = {
    device_destinations.view};
  cugraph_type_erased_device_array_view_t const* edge_id_views[] = {device_edge_ids.view};

  auto status = cugraph_graph_create_mg(c_handle,
                                        &properties,
                                        nullptr,
                                        source_views,
                                        destination_views,
                                        nullptr,
                                        config.edge_ids ? edge_id_views : nullptr,
                                        nullptr,
                                        FALSE,
                                        1,
                                        FALSE,
                                        FALSE,
                                        FALSE,
                                        FALSE,
                                        &graph,
                                        &error);
  check_cugraph(status, error, "cugraph_graph_create_mg");
  if (error != nullptr) {
    cugraph_error_free(error);
    error = nullptr;
  }

  cugraph_rng_state_t* rng_state{nullptr};
  status = cugraph_rng_state_create(
    c_handle, static_cast<uint64_t>(12345 + config.rank), &rng_state, &error);
  check_cugraph(status, error, "cugraph_rng_state_create");
  if (error != nullptr) {
    cugraph_error_free(error);
    error = nullptr;
  }

  cugraph_sampling_options_t* options{nullptr};
  status = cugraph_sampling_options_create(&options, &error);
  check_cugraph(status, error, "cugraph_sampling_options_create");
  if (error != nullptr) {
    cugraph_error_free(error);
    error = nullptr;
  }
  cugraph_sampling_set_with_replacement(options, config.with_replacement ? TRUE : FALSE);
  cugraph_sampling_set_return_hops(options, config.return_hops ? TRUE : FALSE);
  cugraph_sampling_set_renumber_results(options, config.renumber ? TRUE : FALSE);

  auto* fanout_view =
    cugraph_type_erased_host_array_view_create(config.fanout.data(), config.fanout.size(), INT32);

  std::cout << "{\"event\":\"setup\",\"rank\":" << config.rank
            << ",\"vertices\":" << config.vertices
            << ",\"local_edges\":" << host_sources.size()
            << ",\"seeds\":" << host_seeds.size() << ",\"fanout\":\"";
  for (size_t index = 0; index < config.fanout.size(); ++index) {
    if (index > 0) { std::cout << ','; }
    std::cout << config.fanout[index];
  }
  std::cout << "\",\"renumber\":" << (config.renumber ? "true" : "false")
            << ",\"return_hops\":" << (config.return_hops ? "true" : "false")
            << ",\"label_offsets\":" << (config.label_offsets ? "true" : "false")
            << ",\"edge_ids\":" << (config.edge_ids ? "true" : "false") << '}'
            << std::endl;

  barrier_and_sync(handle);

  if (!config.build_only) {
    for (int iteration = -config.warmup; iteration < config.iters; ++iteration) {
      barrier_and_sync(handle);
      auto begin = std::chrono::steady_clock::now();
      cugraph_sample_result_t* result{nullptr};
      status = cugraph_homogeneous_uniform_neighbor_sample(
        c_handle,
        rng_state,
        graph,
        device_seeds.view,
        config.label_offsets ? device_label_offsets.view : nullptr,
        fanout_view,
        options,
        FALSE,
        &result,
        &error);
      check_cugraph(status, error, "cugraph_homogeneous_uniform_neighbor_sample");
      if (error != nullptr) {
        cugraph_error_free(error);
        error = nullptr;
      }
      check_cuda(cudaDeviceSynchronize(), "sample cudaDeviceSynchronize");
      auto end = std::chrono::steady_clock::now();

      auto* majors        = cugraph_sample_result_get_majors(result);
      auto* minors        = cugraph_sample_result_get_minors(result);
      auto* result_edge_ids = cugraph_sample_result_get_edge_id(result);
      auto major_size =
        majors == nullptr ? 0 : cugraph_type_erased_device_array_view_size(majors);
      auto minor_size =
        minors == nullptr ? 0 : cugraph_type_erased_device_array_view_size(minors);
      auto edge_id_size = result_edge_ids == nullptr
                            ? 0
                            : cugraph_type_erased_device_array_view_size(result_edge_ids);
      if (major_size != minor_size) { die("sample result major/minor sizes differ"); }
      if (config.edge_ids && edge_id_size != minor_size) {
        die("sample result is missing requested edge IDs");
      }

      auto seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - begin).count();
      std::cout << "{\"event\":\"sample\",\"rank\":" << config.rank
                << ",\"iter\":" << iteration
                << ",\"warmup\":" << (iteration < 0 ? "true" : "false")
                << ",\"seconds\":" << seconds << ",\"majors\":" << major_size
                << ",\"minors\":" << minor_size << ",\"edge_ids\":" << edge_id_size
                << '}' << std::endl;
      cugraph_sample_result_free(result);
    }
  }

  barrier_and_sync(handle);

  cugraph_type_erased_host_array_view_free(fanout_view);
  cugraph_sampling_options_free(options);
  cugraph_rng_state_free(rng_state);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(c_handle);

  check_nccl(ncclCommDestroy(nccl_communicator), "ncclCommDestroy");
  check_cuda(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
  std::cout << "{\"event\":\"done\",\"rank\":" << config.rank << '}' << std::endl;
  return EXIT_SUCCESS;
}
