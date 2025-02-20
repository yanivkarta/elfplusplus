#ifndef PLATFORM_HELPER_H_
#define PLATFORM_HELPER_H_

#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>
#include <map>
#include <set>
#include <mutex>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <sqlite3.h>

//fill in package data in the sql file.







// YK: added support 4 boolean type:
namespace{
extern "C" void fill_packages_from_system(const char* sql_file);
extern "C" int sqlite3_bind_boolean(sqlite3_stmt *stmt, int n, bool value);
extern "C" bool sqlite3_column_boolean(sqlite3_stmt *stmt, int n);
}

namespace elfpp {

//safe queue:
template<class lock_type, typename queue_data_arg>
class safe_queue {
private:
	mutable lock_type _locker;
	mutable std::mutex _condvar;
	mutable std::unique_lock<std::mutex> _varlock;
	std::queue<queue_data_arg*> _mqueue;
	mutable std::condition_variable _mcond;
public:
	safe_queue();
	virtual ~safe_queue();
	inline void wait() {
		_mcond.wait(_varlock);
		//_mcond.wait(_varlock, [this]{return _mqueue.size();});
	}
	queue_data_arg* pop();
	void push(queue_data_arg*);
	size_t size() const;
};

template<class lock_type, typename queue_data_arg>
safe_queue<lock_type, queue_data_arg>::safe_queue() :
		_varlock(_condvar) {

	std::lock_guard<lock_type> g(_locker);

}

template<class lock_type, typename queue_data_arg>
safe_queue<lock_type, queue_data_arg>::~safe_queue() {

	std::lock_guard<lock_type> g(_locker);

}
template<class lock_type, typename queue_data_arg>
queue_data_arg* safe_queue<lock_type, queue_data_arg>::pop() {
	queue_data_arg* ret = NULL;
	std::lock_guard<lock_type> g(_locker);
	ret = _mqueue.front();
	_mqueue.pop();
	return ret;

}
template<class lock_type, typename queue_data_arg>
void safe_queue<lock_type, queue_data_arg>::push(queue_data_arg* arg) {

	std::lock_guard<lock_type> g(_locker);
	_mqueue.push(arg);
	_mcond.notify_one(); //notify subscribers
}

template<class lock_type, typename queue_data_arg>
size_t safe_queue<lock_type, queue_data_arg>::size() const {
	std::lock_guard<lock_type> g(_locker);
	return _mqueue.size();
}


template<class SINGLETON>
class safe_singleton {

	static SINGLETON* _class;
	static std::recursive_mutex _lock;
public:

	static SINGLETON* get_instance();
	static SINGLETON* create_instance();
	static void release_instance();
	safe_singleton();
	virtual ~safe_singleton();

};

template<class SINGLETON>
safe_singleton<SINGLETON>::safe_singleton() {
	std::cout << "constructor for singleton called...";
}
template<class SINGLETON>
safe_singleton<SINGLETON>::~safe_singleton() {
	std::lock_guard<std::recursive_mutex> lock(_lock);

}

template<class SINGLETON>
SINGLETON* safe_singleton<SINGLETON>::create_instance() {

	std::lock_guard<std::recursive_mutex> lock(_lock);

	if (!_class) {
		std::lock_guard<std::recursive_mutex> lock(_lock);
		_class = (_class == NULL) ? new SINGLETON() : _class;
	}
	return _class;

}

template<class SINGLETON> void safe_singleton<SINGLETON>::release_instance() {
	//std::lock_guard<std::recursive_mutex> lock(_lock);
	if (_class) {
		delete _class;
		_class = NULL;
		return;
	}
}

template<class SINGLETON>
SINGLETON* safe_singleton<SINGLETON>::get_instance() {
	if (!_class) {
		return safe_singleton<SINGLETON>::create_instance();
	}
	return _class;
}

template<class SINGLETON> SINGLETON* safe_singleton<SINGLETON>::_class = NULL;
template<class SINGLETON> std::recursive_mutex safe_singleton<SINGLETON>::_lock;



struct ProcInfo {
	pid_t pid;
	uid_t uid;
	std::string name; //from comm
	std::string fullpath; //from cmdline
};

class proc_helper: public safe_singleton<proc_helper> {
public:
	proc_helper() {
	}
	virtual ~proc_helper() {
	}
	void fill_existing(std::set<pid_t>& existing) const;
	void remove_obsolete(std::set<pid_t>& obsolete);
	void update_info();
	std::string get_full_path(pid_t pid) const;
	std::string get_name(pid_t pid) const;

protected:
	void refresh_process_info(pid_t pid);
	uid_t getuid_of_proc(pid_t pid);

private:
	std::map<pid_t, std::shared_ptr<ProcInfo>> _procs;
	mutable std::recursive_mutex _lock;
	std::map<uint64_t, std::string> _pathlookup; //local path hash cache.
};


//db 


}//namespace elfpp

namespace sqlite {
// YK : edited internal what for the exception code.
class db_error: public std::exception {
	std::string _what;
public:
	db_error(int c) :
			_code(c) {
		_what = sqlite3_errstr(c);
	}
	virtual ~db_error() throw ();

	const char *what() const throw ();

	int code() const {
		return _code;
	}

private:
	int _code;
};

namespace sql {

static inline void check(int c) {
	if (c != SQLITE_OK) {
		throw db_error(c);
	}
}

void destroy_blob(void *blob);
void destroy_text(void *blob);
void ignore_text(void *blob);
}
//
// Database handle
class db {
public:
	db() :
			_db(nullptr) {
	}

	db(const std::string& filename) {
		sqlite::sql::check(::sqlite3_open(filename.c_str(), &_db));
	}

	db(const char *filename) {
		sqlite::sql::check(::sqlite3_open(filename, &_db));
	}

	~db() {
		if (_db)
			::sqlite3_close(_db);
	}

	db(const db&) = delete;
	db& operator=(const db&) = delete;

	/*    void swap(db& r)
	 {
	 std::swap(_db, r._db);
	 }
	 */
	db(db&& r) :
			_db(r._db) {
		r._db = nullptr;
	}

	db& operator=(db&& r) {
		db m(std::move(r));
		std::swap(_db, m._db);
		return *this;
	}

	::sqlite3 *get() {
		return _db;
	}

	const ::sqlite3 *get() const {
		return _db;
	}

	// Number of changes due to the most recent statement.
	unsigned int changes() const {
		return ::sqlite3_changes(_db);
	}

	// Execute a simple statement
	void exec(const std::string& text) {
		sql::check(
				::sqlite3_exec(_db, text.c_str(), nullptr, nullptr, nullptr));
	}

	void exec(const char *text) {
		sql::check(::sqlite3_exec(_db, text, nullptr, nullptr, nullptr));
	}
	std::string check_error_details() {
		std::string ret = "";
		if (_db) {
			ret = sqlite3_errmsg(_db);
		}
		return ret;
	}

private:
	::sqlite3 *_db;
};

// Statement
class stmt {
public:
	stmt() :
			_stmt(nullptr) {
	}

	stmt(db& db, const char *sql) {
        //YK: prepared statement should not deallocate sql string 
        //make sure it will not be deallocated.
		
        sqlite::sql::check(
				::sqlite3_prepare_v2(db.get(), sql, -1, &_stmt, nullptr));
	}

	~stmt() {
		if (_stmt)
			::sqlite3_finalize(_stmt);
	}

	stmt(const stmt&) = delete;
	stmt& operator=(const stmt&) = delete;

	void swap(stmt& r) {
		std::swap(_stmt, r._stmt);
	}

	stmt(stmt&& r) :
			_stmt(r._stmt) {
		r._stmt = nullptr;
	}

	stmt& operator=(stmt&& r) {
		stmt m(std::move(r));
		swap(m);
		return *this;
	}

	::sqlite3_stmt *get() {
		return _stmt;
	}

	const ::sqlite3_stmt *get() const {
		return _stmt;
	}

	class data_binder {
	public:
		data_binder(stmt& s) :
				_stmt(s._stmt) {
		}

		data_binder& blob(unsigned int i, const void *data, size_t len) {
			uint8_t *copy = new uint8_t[len];

            std::copy((uint8_t *) data, (uint8_t *) data + len, copy);
			sqlite::sql::check(
					::sqlite3_bind_blob(_stmt, i, copy, len,
							sqlite::sql::destroy_blob));
			return *this;
		}

		data_binder& blob_ref(unsigned int i, const void *data, size_t len) {
			sqlite::sql::check(
					::sqlite3_bind_blob(_stmt, i, data, len, nullptr));
			return *this;
		}

		data_binder& real(unsigned int i, double value) {
			sqlite::sql::check(::sqlite3_bind_double(_stmt, i, value));
			return *this;
		}

		data_binder& int32(unsigned int i, int32_t value) {
			sqlite::sql::check(::sqlite3_bind_int(_stmt, i, value));
			return *this;
		}

		data_binder& int64(unsigned int i, int64_t value) {
			sqlite::sql::check(::sqlite3_bind_int64(_stmt, i, value));
			return *this;
		}

		data_binder& null(unsigned int i) {
			sqlite::sql::check(::sqlite3_bind_null(_stmt, i));
			return *this;
		}
		data_binder & boolean(unsigned int i, bool value) {
			sqlite::sql::check(::sqlite3_bind_boolean(_stmt, i, value));
			return *this;

		}

		data_binder& text(unsigned int i, const char *orig) {
			
            //YK: text is not null terminated
            //copy it to ensure it is null terminated   
            size_t len = std::string(orig).size();
            sqlite::sql::check(
                    ::sqlite3_bind_text(_stmt, i, orig, len, sqlite::sql::ignore_text));
            return *this;
            

		}

		data_binder& text(unsigned int i, const std::string& value) {
			const char *orig = value.c_str();
			const size_t len = value.size();
			char *copy = (char*)  sqlite3_malloc(len);
            if (!copy) {
                throw std::bad_alloc();
            }
            std::copy(orig, orig + len, copy);

			sqlite::sql::check(
					::sqlite3_bind_text(_stmt, i, copy, len,
							sqlite::sql::destroy_text));
			return *this;
		}

		data_binder& text_ref(unsigned int i, const std::string& value) {
			sqlite::sql::check(
					::sqlite3_bind_text(_stmt, i, value.c_str(), value.size(),
							nullptr));
			return *this;
		}

		data_binder& text_ref(unsigned int i, const char *value) {
			sqlite::sql::check(
					::sqlite3_bind_text(_stmt, i, value, -1, nullptr));
			return *this;
		}

		void clear() {
			sqlite::sql::check(::sqlite3_clear_bindings(_stmt));
		}

	private:
		::sqlite3_stmt *_stmt = nullptr;
	};

	data_binder bind() {
		return data_binder(*this);
	}

	bool step() {

		const int c = ::sqlite3_step(_stmt);

		if (c == SQLITE_ROW)
			return true;

		if (c == SQLITE_DONE)
			return false;

		throw db_error(c);
	}

	void exec() {
		while (step())
			;
	}

	void reset() {
		sqlite::sql::check(::sqlite3_reset(_stmt));

	}

	class reader {
	public:
		reader(stmt& s) :
				_stmt(s._stmt) {
		}

		const void *blob(unsigned int i) {
			return ::sqlite3_column_blob(_stmt, i);
		}

		size_t size(unsigned int i) {
			return ::sqlite3_column_bytes(_stmt, i);
		}

		double real(unsigned int i) {
			return ::sqlite3_column_double(_stmt, i);
		}

		int32_t int32(unsigned int i) {
			return ::sqlite3_column_int(_stmt, i);
		}

		int64_t int64(unsigned int i) {
			return ::sqlite3_column_int64(_stmt, i);
		}

		const char *cstr(unsigned int i) {
			return reinterpret_cast<const char *>(::sqlite3_column_text(_stmt,
					i));
		}

		//Y.K - boolean support.
		bool boolean(unsigned int i) {
			return ::sqlite3_column_boolean(_stmt, i);
		}

		std::string text(unsigned int i) {
			return std::string(cstr(i), size(i));
		}

	private:
		::sqlite3_stmt *_stmt;
	};

	reader row() {
		return reader(*this);
	}

private:
	::sqlite3_stmt *_stmt;
};
}   // namespace sqlite

namespace elfpp {
    


//monitor and parse netlink messages and manage process lists
//
class netlink_helper {

public:

	typedef bool (*on_new_path)(const std::string& path_to_check);
	void register_handler(on_new_path handler) {
		_callbacks.insert(handler);
	}
	void unregister_handler(on_new_path handler) {
		_callbacks.erase(handler);
	}

	netlink_helper();
	bool connect();
	std::string get_fullpath(pid_t pid);
	bool proc_listen(bool);
	void run();
	virtual ~netlink_helper();

private:
	int nl_sock; //netlink socker
	bool _running_;
	proc_helper _helper;
	std::set<on_new_path> _callbacks;

};

class package_helper: public safe_singleton<package_helper> {

public:
	//add to queue:
	void add_to_queue(const std::string& package_hash,const std::string& package_name, const std::string&package_version,const std::string& package_file);
	
private:

	std::string get_package_name(const char* path);

	bool fill_package_hashes(const std::string& package_file,
			const std::string& origfile,
			std::map<std::string, std::string>& _package_hashes);

	bool validate_package_checksums(const std::string& path,
			const std::string& checksum);

protected:
	bool is_white_listed(uint64_t hash);
public:
	explicit package_helper();

	bool is_validated(std::string& path);

	std::map<uint64_t, std::pair<std::string, bool>> white_list;
	std::map<uint64_t, std::map<std::string, std::string>> component_map;
	std::map<uint64_t, uint64_t> path_hash_to_component;
};

// class file_processor
// create hash from path and add to TX queue for hash queries.
//
typedef void (*reg_callback)(const std::string& path, const std::string& hash,
		void* _this);

class file_processor: public elfpp::safe_singleton<file_processor> {
private:
	enum hash_type {
		HASH_MD5, HASH_SHA1, HASH_SHA256, HASH_SHA512
	};
public:
	file_processor();
	virtual ~file_processor();
	void add_path(const std::string& path) {
		std::string* arg = new std::string(path.c_str());
		_rxQ.push(arg);
	}
	//thread callbacks:
	static void process_files(file_processor* _context);
	static void dispatch_requests(file_processor* _context);
	std::string get_hash(const std::string&) const;
	void register_dispatch_handlers(reg_callback callback, void* context);

	typedef struct tagFSEvent {
		std::string filepath;
		std::string hash_data;
		hash_type type; //for now ignoring, using only sha1.
	} file_system_event;
	void stop();
protected:
	bool _running;
	hash_type _default_hash;
	std::thread* _dispatch;
	std::thread* _processing;
	std::map<uint64_t, std::pair<std::string, std::string>> _whitelist;
 	elfpp::safe_queue<std::recursive_mutex, std::string> _rxQ;
	elfpp::safe_queue<std::recursive_mutex, file_processor::file_system_event> _txQ;
	std::vector<std::pair<reg_callback, void*> > _callbacks;

};
std::ostream & operator <<(std::ostream& os, file_processor::tagFSEvent& ev);




class db_dal {

    
public:
	std::string get_global_value(const std::string& key);
	void set_global_value(const std::string& key, const std::string& value);
protected:
	void log_start();
	void log_end();

	virtual ~db_dal()=0;
	explicit db_dal(const std::string& db_file,
			const std::string& component_name);
	std::string component_name;
	std::string db_path;
    std::recursive_mutex _lock;
private:
	//processor
	//timers/timepoints are mutable to be used in const member functions:
	mutable std::chrono::time_point<std::chrono::system_clock> _spoint;
	//not used for now...
	mutable std::chrono::duration<double> elapsed_seconds; //conserve delta of start-end.

};

//implement web related queries .
//for the web component.
class web_dal: virtual public db_dal {
private:
	struct web_batch_task {
	};
//webclient
public:
	enum upload_policy {
		DAILY, HOURLY, EVENT_BASED
	};
	std::string get_api_key();
	bool update_throttle_params(const uint32_t& xlimit, const uint32_t& xused,
			const uint32_t& xremaining, const uint32_t& timetoreset,
			const uint32_t& xinterval, const std::string& limit4 =
					"reputation_api");

	//add path and  signature to checksum queue
	bool add_to_checksum_queue(const std::string& path,
			const std::string& signature);
	//get a queue of paths and checksums
	void get_checksum_queue(
			std::queue<std::pair<std::string, std::string> > & checksum_work);
	//inserts  into upload queue values from checksum queue  where result file is unknown
	void move_results_to_upload_queue();
	// update the results :
	bool update_checksum_results(const std::string& signature,
			const std::string &result, uint32_t resultcode = 0);
	// update the upload results :
	bool update_upload_results(size_t batch_id, const std::string& result_body,
			uint32_t resultcode);
	// mark working pid/tid for each checksum item.
	bool mark_checksum_queue();

	web_dal(const std::string &file);
	virtual ~web_dal();

};
//
//implement file processor dal
class proc_dal: virtual public db_dal {

protected:

	bool update_verified_dist_file(const std::string& path,
			const std::string sha1, const std::string& package_name, //treat empty as dbnull
			bool is_verified = false);

public:

	//if set_verified fails on unique constraints it calls update.
	bool set_verified_dist_file(const std::string& path, const std::string sha1,
			const std::string& package_name, 	//treat empty as dbnull
			bool is_verified = false);

	//fill hash<std::string>(path)<->sha1
	bool fill_white_list_hash(
			std::map<uint64_t, std::pair<std::string, bool>>& hash_);
	//fills the full whitelist path<->sha1, from db.
	bool fill_white_list(
			std::map<std::string, std::pair<std::string, bool>>& map_);
	proc_dal(const std::string& db_path);
	virtual ~proc_dal();
}; //proc_dal




} //namespace elfpp

#endif