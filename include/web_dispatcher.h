#ifndef __WEB_DISPATCHER_H_
#define __WEB_DISPATCHER_H_


#include <elf.h>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>

//mmap/munmap
#include <sys/mman.h>
#include <sys/stat.h>


namespace elfpp {


 class web_dispatcher : public std::enable_shared_from_this<web_dispatcher> {
	  
private:
	std::string db_path;
	std::string component_name;
	std::string component_version;
	std::string component_release;
	size_t rpm;
	size_t cache;
	size_t nqueries;
	size_t nqpm;
	std::chrono::time_point<std::chrono::system_clock> last_dispatch; //not used 
	std::string api_key;
	std::string result_file; //notused
	std::recursive_mutex _lock_unsafe_queue;
	std::queue<std::pair<std::string,std::string>> unsafe_upload_queue;

public:
	web_dispatcher();
	inline void set_api_key(std::string apikey) {
		api_key = apikey;
	}
	void set_cache(size_t nqueries);
	void set_rate(size_t nqpm);
	void parse_header_info(const std::string& header);
	void parse_body_info(const std::string& body);
	std::string build_url(const std::string& hash) const;
	//requests per minute:
	size_t getRPM() const;
	virtual ~web_dispatcher();
	//callback:
	static void on_new_dispatch_task(const std::string& filename,
			const std::string& hash, void* _t);

	//
	std::pair<std::string,std::string> next_upload_file();
	//thread dispatch for file upload[not batch schedule]
	static void dispatch_queue(web_dispatcher* _this);
	//not impl.
	void batch_upload(
			std::chrono::time_point<std::chrono::system_clock>& schedule);

};
} // namespace elfpp

#endif  //