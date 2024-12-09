#include "platform_helper.h"
#ifndef restrict
#define restrict
#endif
 #include <sys/socket.h>
 extern "C" {
#include <linux/netlink.h>
#include <linux/connector.h>
#include <linux/cn_proc.h>	 
#include <gnutls/gnutls.h> //for hashing files use gnutls
#include <gnutls/dane.h> //for hashing files use gnutls
#include <gnutls/crypto.h> //for hashing files use gnutls

 }

#include <stdbool.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h> 
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>
//mmap/munmap
#include <sys/mman.h>

#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <string>

//#include <wget.h>//for gnutls hash_file

#include <fstream> 

extern "C" {

//'adapters' for boolean binding
int sqlite3_bind_boolean(sqlite3_stmt *stmt, int n, bool value) {
	return sqlite3_bind_int(stmt, n, value ? 1 : 0);
}

bool sqlite3_column_boolean(sqlite3_stmt *stmt, int n) {
	return sqlite3_column_int(stmt, n) != 0;
}

 void on_sigint(int unused) {
	//file_processor::get_instance()->stop();
	bool sigint = unused==SIGINT;
	std::cout << "[+] recieved SIGINT exiting gracefully..." << std::endl;
	elfpp::file_processor::get_instance()->stop();

	
	
    elfpp::file_processor::release_instance();
	::exit(unused);

	//double free :)
	

}

}

namespace elfpp {

int
wget_hash_file(const char* algorithm, const char* file, char* digest_text, int digest_text_size); //for gnutls hash_file 

std::string elfpp::netlink_helper::get_fullpath(pid_t pid) {
	return _helper.get_full_path(pid);
}

//db_dal 
//db_dal :
db_dal::db_dal(const std::string& db_file, const std::string& _name) :
		component_name(_name), db_path(db_file), elapsed_seconds() {
	_spoint = std::chrono::system_clock::now();
	log_start();
}
db_dal::~db_dal() {

	std::chrono::time_point<std::chrono::system_clock> now_ =
			std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = now_ - _spoint;
	std::cout << "[+] db_dal destructor for " << this->component_name
			<< " took " << elapsed.count() << "to execute " << std::endl;
}
//returns single K/V entry from global table.
//for settings (e.g. api-keys , etc... )
std::string db_dal::get_global_value(const std::string& key) {
	std::string ret;
	try {

		// create database:
		sqlite::db database_(this->db_path);
		//prepare statement:
		sqlite::stmt rs(database_,
				"SELECT FIELD_VALUE from GENERAL_SETTINGS WHERE FIELD_KEY = ?");
		rs.bind().text(1, key.c_str());
		if (rs.step())
			ret = rs.row().text(0);

	} catch (sqlite::db_error& error) {
		std::cerr << "[-] get_global_value cought: " << error.what() << " ["
				<< error.code() << "]" << std::endl;

	}

	this->elapsed_seconds = std::chrono::system_clock::now() - _spoint;

	return ret;

}

//
void db_dal::set_global_value(const std::string& key,
		const std::string& value) {

	try {
		// create database:
		sqlite::db database_(this->db_path);
		//prepare statement:
		sqlite::stmt s(database_,
				"UPDATE GENERAL_SETTINGS SET FIELD_VALUE='?' FIELD_KEY='?'");
		//bind :
		s.bind().text(1, key.c_str()).text(2, value.c_str());
		//execute
		s.exec();
	} catch (sqlite::db_error& error) {
		std::cerr << "[-] set_global_value cought: " << error.what() << " ["
				<< error.code() << "]" << std::endl;

	}
	this->elapsed_seconds = std::chrono::system_clock::now() - _spoint;

}

//log each component's workload
void db_dal::log_start() {

    std::lock_guard<std::recursive_mutex> lock(_lock);

	try {

		// create database:
		sqlite::db database_(this->db_path);
		//prepare statement:
		sqlite::stmt s(database_,
				"UPDATE LOG_COMPONENT SET RUN_DURATION_START=DATETIME('NOW'),RUN_DURATION_END = NULL WHERE COM_NAME=?");
		//bind :
		s.bind().text(1, this->component_name.c_str());
		//execute
		s.exec();

        // create database:
        //		sqlite::db database_(this->db_path);
        //		//prepare statement:
        //		sqlite::stmt s(database_,
        //				"INSERT INTO LOG_COMPONENT(COM_NAME) VALUES(?)");
        //		//bind :
        //		s.bind().text(1, this->component_name.c_str());
        //		//execute
        //		s.exec();


	} catch (sqlite::db_error& error) {
		std::cout << "[-]log_start cought: " << error.what() << " ["
				<< error.code() << "]" << std::endl;

	}
	this->elapsed_seconds = std::chrono::system_clock::now() - _spoint;

}
//
void db_dal::log_end() {

    std::lock_guard<std::recursive_mutex> lock(_lock);
	try {
		// create database:
		sqlite::db database_(this->db_path);
		//prepare statement:
		sqlite::stmt s(database_,
				"UPDATE LOG_COMPONENT SET   RUN_DURATION_END = DATETIME('NOW' ) WHERE COM_NAME=?");
		//bind :
		s.bind().text(1, this->component_name.c_str());
		s.exec();

	} catch (sqlite::db_error& error) {
		std::cout << "[-]log_end cought: " << error.what() << " ["
				<< error.code() << "]" << std::endl;

	}
	this->elapsed_seconds = std::chrono::system_clock::now() - _spoint;

}

std::string web_dal::get_api_key() {
	std::string ret = db_dal::get_global_value("apikey");
	return ret;
}

bool web_dal::add_to_checksum_queue(const std::string& path,
		const std::string& signature) {
	bool ret = false;
	try {
		sqlite::db database_(this->db_path);
        sqlite::stmt s(database_,
				"INSERT INTO QUERY_BUILDER_CHECKSUM_QUEUE(PATH,SHA1) VALUES(?,?);\n ");
		s.bind().text(1, path.c_str()).text(2, signature.c_str());
		s.exec();
		ret = true;
	} catch (sqlite::db_error& error) {

		std::cout << "proc_dal::set_verified_dist_file cought: " << error.what()
				<< " [" << error.code() << "]" << std::endl;

	}

	//get a queue of paths and checksums
	return ret;
}


//run - blocks, should be on a seperate thread.
void netlink_helper::run() {

	struct cn_msg_non_flex {
	struct cb_id id;

	__u32 seq;
	__u32 ack;

	__u16 len;		/* Length of the following data */
	__u16 flags;
	__u8 *data;
	};	

	int rc;
	struct
		__attribute__ ((aligned(NLMSG_ALIGNTO))) {
			struct nlmsghdr nl_hdr;
			struct
				__attribute__ ((__packed__)) {
					struct cn_msg_non_flex cn_msg;
					struct proc_event proc_ev;
				};
			} nlcn_msg;

	while (_running_) {
		//read :
		rc = recv(nl_sock, &nlcn_msg, sizeof(nlcn_msg), 0);
		if (rc <=0 ) {
			if ((errno == EINTR) || (errno == ENOBUFS)) {
				//queue is full, wait.
				usleep(1000);
				continue;
			}

			std::cout << "[-]netlink recv error " << errno << std::endl;
				break;
			}
		//success:
		if (nlcn_msg.proc_ev.what == PROC_EVENT_EXEC) {
					std::string fullpath = get_fullpath(
							nlcn_msg.proc_ev.event_data.exec.process_pid);

					if (fullpath.length() > 3) {

						if (nlcn_msg.proc_ev.event_data.exec.process_tgid
								!= nlcn_msg.proc_ev.event_data.exec.process_pid)
							std::cout << "tgid="
									<< nlcn_msg.proc_ev.event_data.exec.process_tgid
									<< "pid="
									<< nlcn_msg.proc_ev.event_data.exec.process_pid
									<< " " << ""
									<< nlcn_msg.proc_ev.event_data.exec.process_pid
									<< " " << fullpath << std::endl;

						//update info if processes were added/removed.
						_helper.update_info();

					}
					//		std::cout<<"[+]exec: ignoring "<<this->_helper.get_name(nlcn_msg.proc_ev.event_data.exec.process_pid)<<std::endl;

				} else {

					//we don't care about fork/etc..
					//if the pid_t list needs updates:
					// update  existing proc info with :
					// _helper.update_info();
 					//	currently we do nothing and wait only for proc_event::PROC_EVENT_EXEC
					//  can extend it later with the rest of the enum
					//	and corresponding structs for delete/fork/change uid/gid/etc...

				}
			}
		}

		//use netlink socket connector
		bool netlink_helper::proc_listen(bool enable) {
			int rc;

			if (nl_sock == -1) {
				return false;
			}

			
			struct __attribute__ ((aligned(NLMSG_ALIGNTO))) {
					struct nlmsghdr nl_hdr;
					struct
						__attribute__ ((__packed__)) {
							struct cn_msg cn_msg;
							//avoid padding between cn_msg and cn_mcast:
							//enum proc_cn_mcast_op cn_mcast; 
						};
					} nlcn_msg;


					memset(&nlcn_msg, 0, sizeof(nlcn_msg));
					nlcn_msg.nl_hdr.nlmsg_len = sizeof(nlcn_msg);
					nlcn_msg.nl_hdr.nlmsg_pid = getpid();
					nlcn_msg.nl_hdr.nlmsg_type = NLMSG_DONE;

					nlcn_msg.cn_msg.id.idx = CN_IDX_PROC;
					nlcn_msg.cn_msg.id.val = CN_VAL_PROC;
					nlcn_msg.cn_msg.len = sizeof(enum proc_cn_mcast_op);
					//nlcn_msg.cn_mcast =
					//		enable ?
					//				PROC_CN_MCAST_LISTEN : PROC_CN_MCAST_IGNORE;

					rc = send(nl_sock, &nlcn_msg, sizeof(nlcn_msg), 0);
					if (rc == -1) {
						perror("netlink send");
						return false;
					}

					return true;
				}
				netlink_helper::~netlink_helper() {
					if (nl_sock > 0)
						close(nl_sock);
					nl_sock = -1;
				}
				netlink_helper::netlink_helper() :
						nl_sock(-1), _running_(true) {

					int rc;
					struct sockaddr_nl sa_nl;

					nl_sock = socket(PF_NETLINK, SOCK_DGRAM, NETLINK_CONNECTOR);
					if (nl_sock == -1) {
						throw std::runtime_error(
								"can't open NETLINK_CONNECTOR socket, must be root");
					}

					sa_nl.nl_family = AF_NETLINK;
					sa_nl.nl_groups = CN_IDX_PROC;
					sa_nl.nl_pid = getpid();

					rc = bind(nl_sock, (struct sockaddr *) &sa_nl,
							sizeof(sa_nl));
					if (rc == -1) {

						close(nl_sock);
					}
					_helper.update_info();

				}
				//removes old paths
				void proc_helper::remove_obsolete(std::set<pid_t>& obsolete) {
					std::lock_guard<std::recursive_mutex> lock(_lock);
					for (auto o : obsolete)
						_procs.erase(o);

				}
				//get comm name from pid
				std::string proc_helper::get_name(pid_t pid) const {
					char path[4096] = { '\0' }, buf[4096] = { '\0' };
					std::string retval;
					FILE* cmd = NULL;
					std::string data;
					snprintf(path, 4096, "/proc/%d/comm", pid);
					cmd = fopen(path, "r");
					if (cmd) {
						fscanf(cmd, "%4095s", buf);
						fclose(cmd);
						retval = buf;
					}
					return retval;

				}
				//refresh the internal pid list
				void proc_helper::refresh_process_info(pid_t pid) {

					std::lock_guard<std::recursive_mutex> lock(_lock);
					bool bnew = false;
					auto pidDataIt = _procs.find(pid);
					std::shared_ptr<ProcInfo> pidData;
					if (pidDataIt == _procs.end()) {
						pidData = std::make_shared<ProcInfo>();
						_procs.insert(
								std::pair<pid_t, std::shared_ptr<ProcInfo>>(pid,
										pidData));
						bnew = true;
					} else {
						bnew = false;
						pidData = pidDataIt->second;
					}
					auto originalPidData = std::make_shared<ProcInfo>(*pidData);
					pidData->uid = getuid_of_proc(pid);
					pidData->name = get_name(pid);
					pidData->fullpath = get_full_path(pid);

					// if it's a new pid and the path is valid
					// add to  internal path hash lookup
					// and send to processing queue
					//
					if (bnew && pidData->fullpath.size()) {
						std::hash<std::string> string_hash;
						uint64_t hash = string_hash(pidData->fullpath);
						auto iter = _pathlookup.find(hash);

						if (iter == _pathlookup.end()) {
							//insert new path
							std::cout << "[+] new path : " << pidData->fullpath
									<< std::endl;
							_pathlookup.insert(
									std::make_pair(
											string_hash(pidData->fullpath),
											pidData->fullpath));

                            elfpp::file_processor::get_instance()->add_path(
									pidData->fullpath);

						}

					}
				}

				std::string proc_helper::get_full_path(pid_t pid) const {
					char path[4096] = { '\0' }, buf[4096] = { '\0' };
					//if everything fails return name as path.
					std::string retval;
					FILE* cmd = NULL;
					std::string data;

					//we read the buffer, we have the default value,
					//let's be more precise if possible:

					snprintf(path, 4096, "/proc/%d/exe", pid);
					int ret = readlink(path, buf, 4096);
					if (ret == -1 || strnlen(buf, 4096) < 3) {
						//override with cmdline instead if it works:
						snprintf(path, 4096, "/proc/%d/cmdline", pid);
						std::string cmdline;
						cmd = fopen(path, "r");
						if (cmd) {
							ret = fscanf(cmd, "%4095s", buf);
							retval = buf;
							fclose(cmd);

						} else {
							retval = get_name(pid);
						}

					} else {
						//std::cerr << "*" ;
						retval = buf;
					}
					return retval;

				}

				//read /proc/%d/ dir.
				void proc_helper::update_info() {
					std::lock_guard<std::recursive_mutex> lock(_lock);
					std::set<pid_t> obsolete;
					fill_existing(obsolete);
					DIR *procDir;
					procDir = opendir("/proc");
					if (!procDir) {
						perror("Failed to open /proc dir");	//throw exception here.
						return;
					}
					struct dirent *pidDir;
					pid_t pid;
					while ((pidDir = readdir(procDir))) {

						if (!isdigit(pidDir->d_name[0])) {
							continue;
						}
						pid = atoi(pidDir->d_name);
						//update existing or add new.
						refresh_process_info(pid);
						if (obsolete.size())
							obsolete.erase(pid);
					}
					if (obsolete.size())
						remove_obsolete(obsolete);
					closedir(procDir);

				}

				void proc_helper::fill_existing(std::set<pid_t>& fill) const {
					std::lock_guard<std::recursive_mutex> lock(_lock);
					for (auto it = _procs.begin(); it != _procs.end(); ++it) {
						fill.insert(it->first);
					}
				}
				uid_t proc_helper::getuid_of_proc(pid_t pid) {
					char filename[64];
					char line[1024];
					unsigned int uid;
					FILE *file;

					sprintf(filename, "/proc/%d/status", pid);
					file = fopen(filename, "r");
					if (!file) {
						return -1;
					}
					while (fgets(line, 1024, file)) {
						sscanf(line, "Uid: %u", &uid);
					}
					fclose(file);
					return uid;

				}

				//
				//platform package helper:
				bool package_helper::validate_package_checksums(
						const std::string& path, const std::string& md5sum) {

					char digest_text[1024] = { };
					std::string lchecksum;

					int err = wget_hash_file("md5", path.c_str(), digest_text,
							1024);
					if (err) {
						std::cerr << "wget_hash_file error returned :" << err
								<< std::endl;
					} else {
						lchecksum = digest_text;

					}
					//not safe:
					//should always convert to byte format.
					return (lchecksum.compare(md5sum) == 0);

				}

				//returns dpkg-query -S //etc...
				std::string package_helper::get_package_name(const char* path) {
					std::string result_string = "";
					int rc = EXIT_SUCCESS;
					int fds[2];
					rc = pipe(fds);
					if (rc != -1) {
						pid_t task = fork();
						if (task == -1) {
							rc = task;
							std::cout << "[-] fork failed ..." << std::endl;

						} else if (task == 0) {

							//child executes process...

							while (-1 == dup2(fds[1], STDOUT_FILENO)
									&& errno == EINTR) {
								usleep(0);
							};
							while (-1 == dup2(fds[1], STDERR_FILENO)
									&& errno == EINTR) {
								usleep(0);
							};

							close(fds[1]);
							close(fds[0]);

							char* params[] = { const_cast<char*>("dpkg-query"),
									const_cast<char*>("--search"),
									const_cast<char*>(path), nullptr };
							rc = execvp("/usr/bin/dpkg-query", params);

						} else //parent read:
						{
							char buffer[4096 * 4];
							ssize_t count;

							do {
								count = read(fds[0], buffer, sizeof(buffer));
							} while (count == -1 && errno == EINTR);
							if (count == -1) {
								std::cerr << "[-] error reading stream..."
										<< std::endl;

							} else {
								char result[4096] = { '\0' }, result1[4096] = {
										'\0' };
								sscanf(buffer, "%4095s:%4095s", result,
										result1);
								ssize_t nlen = strnlen(result, 4095);

								if (nlen > 2) //truncate ':'
									result[nlen - 1] = '\0';

								result_string = result;
								if (result_string.compare("dpkg-query") == 0) {
									result_string = "";
								} else
									std::cout << "[+]  component found: :"
											<< std::string(result) << std::endl;

							}
							//	close(fds[0]);
							//	wait(0);
						}
						close(fds[0]);
						close(fds[1]);
						close(rc);
						//to avoid defunct state
						//when dpkg doesn't find a package
						::kill(task, 15);
					}
					return result_string;
				}

				//parses /var/lib/dpkg/info/%s.md5sum :
				bool package_helper::fill_package_hashes(
						const std::string& package_file,
						const std::string& orig_file,
						std::map<std::string, std::string>& _package_hashes) {
					bool ret = false;

					std::string file_path = "/var/lib/dpkg/info/";
					file_path += package_file;
					file_path += ".md5sums";
					std::cout << "[+] opening " << file_path << " "
							<< std::endl;
					std::ifstream in_stream(file_path.c_str(), std::ios::in);
					while (in_stream.good()) {
						std::string key, value;
						in_stream >> value >> key;

						if (key.length() && key[0] != '/')
							key = '/' + key;
						//	std::cout<<"[+] key:" <<key<<"[value]:"<<value << std::endl;
						_package_hashes.insert(std::make_pair(key, value));
						if (key.compare(orig_file) == 0) {
							ret = true;
							std::cout
									<< "[+]package file hash found, indexing for caching.";	//

						}
					}
					return ret;
				} //fill_package_hashes

				bool elfpp::package_helper::is_validated(std::string& path) {
					//check cache :

					//FIXME:

					//for now ignore everything and return true if it's in the whitelist
					//future work will also validate sha1 on db against file.

					uint64_t path_hash = std::hash<std::string>()(path);
					bool bwl = is_white_listed(path_hash);
					bool validated = false;

					if (bwl)
						return bwl;

					if (this->path_hash_to_component.find(path_hash)
							!= std::end(path_hash_to_component)) {
						//we already know a component is mapped to this path, skip.
						//it was not whitelisted, but it was previously checked.
						return true;
					} else {

						//get component name:
						auto pkg = this->get_package_name(path.c_str());
						if (pkg.length() < 2)
							return false;
						// have we seen this package?
						uint64_t package_key = std::hash<std::string>()(pkg);

						if (this->component_map.find(package_key)
								!= component_map.end()) {

							if (component_map[package_key].find(path)
									!= std::end(component_map[package_key])) {
								//should never get here, let's validate it anyways
								return validate_package_checksums(path,
										component_map[package_key][path]);//let's ignore it anyway for now.
							} else {
								//we should never get here
								return false;
							}

						} // if we didn't find it, we probably we should  load it:

						std::map<std::string, std::string> hashes;

						if (this->fill_package_hashes(pkg, path, hashes)) {
							std::cout << "[+] updated component hashes for "
									<< pkg << std::endl;
							component_map.insert(
									std::make_pair(package_key, hashes));

							if (validate_package_checksums(path,
									component_map[package_key][path])) {
								validated = true;
								std::cout << "[+] path : " << path
										<< " passed validation succesfully!"
										<< std::endl;

								this->path_hash_to_component.insert(
										std::make_pair(
												std::hash<std::string>()(path),
												std::hash<std::string>()(pkg)));
							} else {
								std::cerr
										<< "[---][-----] Failed to validate checksum.... consider uploading "
										<< std::endl;
							}
							//add to database :
							//
							//--------------------------
							//TODO: -> change db paths
							{
								//open-insert-close.
								std::string sha1 =
										file_processor::get_instance()->get_hash(
												path);
								elfpp::proc_dal dal("elfpp.db");
								dal.set_verified_dist_file(path, sha1, pkg,
										validated);
								this->white_list.insert(
										std::make_pair(path_hash,
												std::make_pair(sha1,
														validated)));

							}

						}

					}

					return validated;

				}

				bool package_helper::is_white_listed(uint64_t process) {
					//search keys for uint64_t hash of path  white list:
					if (white_list.find(process) != white_list.end()) {
						//return is_verified :)
						return white_list[process].second;

					}
					return false;
				}
				package_helper::package_helper() :
						safe_singleton<package_helper>() {
					elfpp::proc_dal dal("elfpp.db");

					if (dal.fill_white_list_hash(white_list)) {

						std::cout << "[+] updated white list with "
								<< white_list.size() << " entries" << std::endl;

					}


                } // package_helper::package_helper

				//
				//platform package helper:
 				//returns dpkg-query -S //etc...

			void package_helper::add_to_queue(const std::string& package_hash,const std::string& package_name, const std::string&package_version,const std::string& package_file)
			{

					//use dal to add to queue 
				try{
					elfpp::proc_dal dal("elfpp.db");
					dal.set_verified_dist_file(package_file, package_hash, package_name, true); 

				}	
				catch(sqlite::db_error& error){
					std::cout << "package_helper::add_to_queue cought: " << error.what() << std::endl;
				}	


			}



//fill hash<std::string>(path)<->sha1
bool proc_dal::fill_white_list_hash(
		std::map<uint64_t, std::pair<std::string, bool>>& hash_) {
	try {
		sqlite::db database_(this->db_path);
		sqlite::stmt s(database_,
				"SELECT  PATH,SHA1,OS_PACKAGE,DIST_VALIDATED  FROM  VERIFIED_WHITE_LIST ;\n ");

		bool validated;
		std::string path, sha1, os_pack;
		while (s.step()) {
			path = s.row().text(0);
			sha1 = s.row().text(1);
			os_pack = s.row().text(2);
			validated = s.row().boolean(3);
			hash_.insert(
					std::make_pair(std::hash<std::string>()(path),
							std::make_pair(sha1, validated)));

		}
		return true;
	} catch (sqlite::db_error& error) {
		std::cout << "proc_dal::set_verified_dist_file cought: " << error.what()
				<< " [" << error.code() << "]" << std::endl;

	}
	return false;
}


bool proc_dal::update_verified_dist_file(const std::string& path,
		const std::string sha1, const std::string& package_name,//treat empty as dbnull
		bool is_verified) {
	//todo: update change of sha1/verified state on  a different table,
	//		for event logging purposes.
	bool ret;
	try {
		sqlite::db database_(this->db_path);
		sqlite::stmt s(database_,
				"UPDATE VERIFIED_WHITE_LIST SET SHA1=?,OS_PACKAGE=?,DIST_VALIDATED=? WHERE PATH=?  ");
		s.bind().text(1, sha1.c_str()).text(2, package_name.c_str()).boolean(3,
				is_verified).text(4, path.c_str());
		s.exec();
		ret = true;
	} catch (sqlite::db_error& error) {
		std::cout << "[+]proc_dal::update_verified_dist_file cought: "
				<< error.what() << " [" << error.code() << "]" << std::endl;

		ret = false;
	}
	return ret;
}
bool proc_dal::set_verified_dist_file(const std::string& path,
		const std::string sha1, const std::string& package_name,//treat empty as dbnull
		bool is_verified) {
	try {
		sqlite::db database_(this->db_path);
		sqlite::stmt s(database_,
				"INSERT INTO VERIFIED_WHITE_LIST(PATH,SHA1,OS_PACKAGE,DIST_VALIDATED) VALUES(?,?,?,?);\n ");
		s.bind().text(1, path.c_str()).text(2, sha1.c_str()).text(3,
				package_name.c_str()).boolean(4, is_verified);
		s.exec();
	} catch (sqlite::db_error& error) {

		if (error.code() == SQLITE_CONSTRAINT_UNIQUE || error.code() == 19) {
			return update_verified_dist_file(path, sha1, package_name,
					is_verified);
		} else {
			std::cout << "proc_dal::set_verified_dist_file cought: "
					<< error.what() << " [" << error.code() << "]" << std::endl;
		}
		return false;
	}
	return true;
}


//fills the full whitelist path<->sha1, from db.
bool proc_dal::fill_white_list(
		std::map<std::string, std::pair<std::string, bool>>& map_) {
	return false;
}
proc_dal::proc_dal(const std::string& db_path) :
		db_dal(db_path, "processor") {

}
proc_dal::~proc_dal() {

}   



} // namespace elfpp


namespace sqlite {
//

db_error::~db_error() throw () {

}

const char *db_error::what() const throw () {
	return _what.c_str();
}

void sqlite::sql::destroy_blob(void *blob) {
	delete[] reinterpret_cast<uint8_t *>(blob);
}

void sqlite::sql::destroy_text(void *blob) {

	delete[] reinterpret_cast<char *>(blob);
}

void sqlite::sql::ignore_text(void *blob) {
    
    //don't delete:
    return;

}

}


namespace elfpp {
    

int
wget_hash_file(const char* algorithm, const char* file, char* digest_text, int digest_text_size)
{
    //use gnutls hash_file,transform algorithm and digest to byte format and return it in digest_text buffer of size digest_text_size bytes      
    //https://gnutls.org/manual/gnutls-hashing.html 
    int results = 0;
    uint8_t digest_max[1024];
    //gnutls_hash_algorithm_t hash_algorithm; 
    gnutls_digest_algorithm_t digest_algorithm;
    if (strcasecmp(algorithm, "md5") == 0) {
        digest_algorithm = GNUTLS_DIG_MD5;
    } else if ((strcasecmp(algorithm, "sha1") == 0)||strcasecmp(algorithm, "sha-1") == 0) { 
        digest_algorithm = GNUTLS_DIG_SHA1;
    } else if (strcasecmp(algorithm, "sha256") == 0) {
        digest_algorithm = GNUTLS_DIG_SHA256;
    } else if (strcasecmp(algorithm, "sha512") == 0) {
        digest_algorithm = GNUTLS_DIG_SHA512;
    } else {
        results = -1;
        std::cout << "unknown algorithm:[" << algorithm << "]" << "returning -1" << std::endl; 
        return results;
    }   

    //hash_file(algorithm, file, digest_text, digest_text_size); 

    //transform algorithm and digest to byte format and return it in digest_text buffer of size digest_text_size bytes   
    gnutls_hash_hd_t dig = {0}; // zero-initialize
    gnutls_hash_init(&dig, digest_algorithm);

    //open ,mmap , hash the file and close it 
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
        std::cerr << "[-]failed to open file:[" << file << "]" << "returning -1" << std::endl;
        results = -1;
        return results;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "[-]failed to stat file:[" << file << "]" << "returning -1" << std::endl; 
        results = -1;
        return results;
    }

    char *buf = (char *) mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) {
        std::cerr << "[-]failed to mmap file:[" << file << "]" << "returning -1" << std::endl;
        results = -1;
        return results;
    }
    results = gnutls_hash(dig, buf, st.st_size);


    if (results < 0) {
        std::cerr << "[-]failed to hash file:[" << file << "]" << "returning -1" << std::endl;
        results = -1;
        return results;
    }

    //copy the digest to the output buffer
    



    //gnutls_hash_get(dig, digest_text, digest_text_size);
    
    munmap(buf, st.st_size);
    close(fd);
    
    gnutls_hash_deinit(dig, digest_max);

    
    //gnutls_hash_deinit(dig);


    //copy the digest to the output buffer
    memcpy(digest_text, digest_max, digest_text_size);

    std::cout << std::endl;
    return results;

}

file_processor::file_processor() :
		safe_singleton<file_processor>(), _running(true), _default_hash(
				file_processor::HASH_SHA1) {
	//before starting the new threads, populate the whitelist
	this->_dispatch = new std::thread(file_processor::dispatch_requests, this);
	this->_processing = new std::thread(file_processor::process_files, this);

}

void file_processor::dispatch_requests(file_processor* context) {
	while (context->_running) {

		//dispatch.
		while (context->_txQ.size() > 0) {

			file_system_event* event = context->_txQ.pop();
			if (event) //although it's unlikely
			{
				for (std::pair<reg_callback, void*> cb : context->_callbacks)
					cb.first(event->filepath, event->hash_data, cb.second);

				delete event;
			}
		}
		context->_txQ.wait();
	}
}

void file_processor::process_files(file_processor* context) {
	std::cerr << "[+]" << " file_processor is waiting for new files ."
			<< std::endl;
	while (context->_running) {

		while (context->_rxQ.size() > 0) {

			std::cout << "\t[+] dequeued:" << context->_rxQ.size() << " files "
					<< std::endl;
			std::string* ppath = context->_rxQ.pop();
			//checking file is whitelisted:

			if (ppath) {

				//check cache hit:
				if (!(package_helper::get_instance()->is_validated(
						*ppath))) {
					// hash and add to txQ:
					file_system_event* pevent = new file_system_event;
					pevent->hash_data = context->get_hash(*ppath);	//get hash
					pevent->filepath = ppath->c_str();			//copy path
					//push event
					if (pevent->hash_data.length() > 1) {
						context->_txQ.push(pevent);
					}
				} else {

					std::cout << "[=] process whitelisted (validated) : "
							<< *ppath << std::endl;
				}

				delete ppath;

			}

			//call event_analyzer with the file system event.
			//-----------------------------------------------
		}
		std::cerr << "[-]" << " file_processor is waiting for new files ."
				<< std::endl;
		context->_rxQ.wait();
	}
}

std::string file_processor::get_hash(const std::string& file) const {
	std::string ret = "";
	char digest_text[1024] = { };
	int err = wget_hash_file("sha-1", file.c_str(), digest_text, 1024);
	if (err) {
		std::cerr << "wget_hash_file error returned :" << err << std::endl;
	}

	ret = digest_text;
	return ret;
}
void file_processor::stop() {
	_running = false;
}
//simple callback and context pointers.
void file_processor::register_dispatch_handlers(reg_callback callback,
		void* context) {
	this->_callbacks.push_back(std::make_pair(callback, context));
}
file_processor::~file_processor() {
	if (_processing) {
		_processing->detach();
		delete _processing;
	}
	_processing = nullptr;
	if (_dispatch) {
		_dispatch->detach();
		delete _dispatch;

	}

	_dispatch = nullptr;

	if (_running)
		_running = false;

}

}//namespace elfpp


 inline bool parse_package_from_line(const std::string& line,
			std::string& package_name, std::string& package_file,
			std::string& package_version, std::string& package_hash) {

		package_name = "";
		package_file = "";
		package_version = "";
		package_hash = "";
		bool ret = false;
		std::istringstream in_stream(line);
		std::string token;
		while (std::getline(in_stream, token, ' ')) {

			if (token.compare("Package:") == 0) {	

				std::getline(in_stream, package_name, '\n');
				continue;	

			} else if (token.compare("Version:") == 0) {	

				std::getline(in_stream, package_version, '\n');
				continue;

			} else if (token.compare("Installed-Size:") == 0) {	

				std::getline(in_stream, package_hash, '\n');
				continue;

			} else if (token.compare("Filename:") == 0) {		

				std::getline(in_stream, package_file, '\n');
				continue;

			}
			
		} //while
		package_hash = package_hash.substr(0, package_hash.find(' ')); 
		if(package_hash.length() > 0)
			ret = true;
		return ret;


		
	}//parse_package_from_line

namespace{
 extern "C" void fill_packages_from_system(const char* sql_file) {

	sqlite::db db(sql_file);
	
	//open file "/var/lib/dpkg/status" and parse packages :

	//read /proc/%d/ dir.
	
	std::ifstream in_stream("/var/lib/dpkg/status");
	std::string line;
	while (std::getline(in_stream, line)) {
		
		std::string package_name;
		std::string package_file;
		std::string package_version;
		std::string package_hash;
		if (!parse_package_from_line(line, package_name, package_file,	
				package_version, package_hash)) {

			continue;
		}
		  
		//parse package hashes. check if validated :
		if (elfpp::package_helper::get_instance()->is_validated(package_hash)) {	// 
			continue;
		} else {
			//not validated.
			//add to queue.
			elfpp::package_helper::get_instance()->add_to_queue(package_hash, package_name, package_version, package_file); 
			
		}
				

	}			

}//fill_packages_from_system
}