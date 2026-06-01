package com.pocketrag.network

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

// ── API 数据模型 ──

data class AskRequest(val query: String)

data class AskResponse(
    val query: String,
    val answer: String,
    val intent: String,
    val sources: List<Map<String, Any>>,
    val latency_ms: Double,
    val cache_hit: Boolean
)

data class MemoryStats(
    val total_documents: Int,
    val collection_name: String,
    val date_range: String?,
    val top_contacts: List<String>
)

// ── Retrofit API 接口 ──

interface PocketMemoryApi {
    @POST("/ask")
    suspend fun ask(@Body request: AskRequest): AskResponse

    @GET("/memory/stats")
    suspend fun getStats(): MemoryStats

    @GET("/health")
    suspend fun health(): Map<String, Any>
}

// ── API 客户端单例 ──

object ApiClient {
    // 默认连接 PC 上的 FastAPI 服务
    // 真机使用时改为 PC 的局域网 IP
    private const val DEFAULT_BASE_URL = "http://10.0.2.2:8000"

    private var baseUrl: String = DEFAULT_BASE_URL

    private var retrofit: Retrofit? = null
    private var api: PocketMemoryApi? = null

    fun setBaseUrl(url: String) {
        if (url != baseUrl) {
            baseUrl = url
            retrofit = null
            api = null
        }
    }

    fun getApi(): PocketMemoryApi {
        if (api == null) {
            retrofit = Retrofit.Builder()
                .baseUrl(baseUrl)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
            api = retrofit!!.create(PocketMemoryApi::class.java)
        }
        return api!!
    }
}
